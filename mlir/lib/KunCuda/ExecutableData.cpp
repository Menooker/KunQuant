//===- ExecutableData.cpp - serialize kun_cuda executable artifacts -------===//
//
// Stores ExecutableData as a JSON metadata file plus a sibling cubin binary.
// The public API is name-based: callers provide only a directory and artifact
// name, and the implementation owns the `<name>.json` / `<name>.cubin`
// convention.
//
//===----------------------------------------------------------------------===//

#include "KunCuda/Runtime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kun_cuda {
namespace {

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

constexpr const char *kFormat = "kun_cuda_executable_data";
constexpr int64_t kVersion = 1;

static void validateArtifactName(const std::string &name) {
  if (name.empty())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: artifact name must be non-empty");
  if (name.find('/') != std::string::npos ||
      name.find('\\') != std::string::npos)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: artifact name must not contain path "
        "separators");
}

static std::string joinArtifactPath(const std::string &dir,
                                    const std::string &fileName) {
  fs::path path(dir);
  path /= fileName;
  return path.string();
}

static std::string jsonFileName(const std::string &name) {
  return name + ".json";
}

static std::string cubinFileName(const std::string &name) {
  return name + ".cubin";
}

static void ensureDirectory(const std::string &dir) {
  if (dir.empty())
    return;

  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to create directory '" + dir +
        "': " + ec.message());
}

static void writeTextFile(const std::string &path, const std::string &text) {
  std::ofstream os(path, std::ios::binary);
  if (!os)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to open '" + path +
        "' for writing");
  os.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!os)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to write '" + path + "'");
}

static void writeBinaryFile(const std::string &path,
                            const std::vector<char> &bytes) {
  std::ofstream os(path, std::ios::binary);
  if (!os)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to open '" + path +
        "' for writing");
  if (!bytes.empty())
    os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!os)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to write '" + path + "'");
}

static std::string readTextFile(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to open '" + path +
        "' for reading");
  std::string text((std::istreambuf_iterator<char>(is)),
                   std::istreambuf_iterator<char>());
  if (!is.eof() && !is)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to read '" + path + "'");
  return text;
}

static std::vector<char> readBinaryFile(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to open '" + path +
        "' for reading");
  std::vector<char> bytes((std::istreambuf_iterator<char>(is)),
                          std::istreambuf_iterator<char>());
  if (!is.eof() && !is)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to read '" + path + "'");
  return bytes;
}

static const char *toString(Datatype dtype) {
  switch (dtype) {
  case Datatype::Float:
    return "f32";
  case Datatype::Double:
    return "f64";
  }
  throw std::runtime_error("kun_cuda::ExecutableData: unknown datatype");
}

static Datatype parseDatatype(const std::string &text,
                              const std::string &jsonPath) {
  if (text == "f32")
    return Datatype::Float;
  if (text == "f64")
    return Datatype::Double;
  throw std::runtime_error(
      "kun_cuda::ExecutableData: unsupported dtype '" + text +
      "' in '" + jsonPath + "'");
}

static const char *toString(KernelKind kind) {
  switch (kind) {
  case KernelKind::Jit:
    return "jit";
  case KernelKind::ExtCsRankF32:
    return "ext_cs_rank_f32";
  case KernelKind::ExtCsRankF64:
    return "ext_cs_rank_f64";
  case KernelKind::ExtCsScaleF32:
    return "ext_cs_scale_f32";
  case KernelKind::ExtCsScaleF64:
    return "ext_cs_scale_f64";
  }
  throw std::runtime_error("kun_cuda::ExecutableData: unknown kernel kind");
}

static KernelKind parseKernelKind(const std::string &text,
                                  const std::string &jsonPath,
                                  const std::string &fieldPath) {
  if (text == "jit")
    return KernelKind::Jit;
  if (text == "ext_cs_rank_f32")
    return KernelKind::ExtCsRankF32;
  if (text == "ext_cs_rank_f64")
    return KernelKind::ExtCsRankF64;
  if (text == "ext_cs_scale_f32")
    return KernelKind::ExtCsScaleF32;
  if (text == "ext_cs_scale_f64")
    return KernelKind::ExtCsScaleF64;
  throw std::runtime_error(
      "kun_cuda::ExecutableData: unsupported kernel kind '" + text +
      "' at " + fieldPath + " in '" + jsonPath + "'");
}

static const json &requireObject(const json &value,
                                 const std::string &jsonPath,
                                 const std::string &fieldPath) {
  if (!value.is_object())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: expected object at " + fieldPath +
        " in '" + jsonPath + "'");
  return value;
}

static const json &requireField(const json &object,
                                const std::string &jsonPath,
                                const std::string &fieldPath,
                                const char *fieldName) {
  requireObject(object, jsonPath, fieldPath);
  auto it = object.find(fieldName);
  if (it == object.end())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: missing field " + fieldPath + "." +
        fieldName + " in '" + jsonPath + "'");
  return *it;
}

static std::string getString(const json &value,
                             const std::string &jsonPath,
                             const std::string &fieldPath) {
  if (!value.is_string())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: expected string at " + fieldPath +
        " in '" + jsonPath + "'");
  return value.get<std::string>();
}

static int64_t getInt64(const json &value,
                        const std::string &jsonPath,
                        const std::string &fieldPath) {
  if (!value.is_number_integer())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: expected integer at " + fieldPath +
        " in '" + jsonPath + "'");
  return value.get<int64_t>();
}

static std::vector<std::string>
getStringArray(const json &value, const std::string &jsonPath,
               const std::string &fieldPath) {
  if (!value.is_array())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: expected array at " + fieldPath +
        " in '" + jsonPath + "'");

  std::vector<std::string> result;
  result.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i)
    result.push_back(getString(value[i], jsonPath,
                               fieldPath + "[" + std::to_string(i) + "]"));
  return result;
}

static std::unordered_map<std::string, int64_t>
getStringIntMap(const json &value, const std::string &jsonPath,
                const std::string &fieldPath) {
  if (!value.is_object())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: expected object at " + fieldPath +
        " in '" + jsonPath + "'");

  std::unordered_map<std::string, int64_t> result;
  for (auto it = value.begin(); it != value.end(); ++it) {
    const std::string path = fieldPath + "." + it.key();
    result.emplace(it.key(), getInt64(it.value(), jsonPath, path));
  }
  return result;
}

static json toJSONArray(const std::vector<std::string> &strings) {
  json array = json::array();
  for (const std::string &s : strings)
    array.push_back(s);
  return array;
}

static json toJSON(const KernelMeta &kernel) {
  json obj = json::object();
  obj["name"] = kernel.kernelName;
  obj["kind"] = toString(kernel.kind);
  obj["inputs"] = toJSONArray(kernel.inputNames);
  obj["outputs"] = toJSONArray(kernel.outputNames);
  obj["unreliable_count"] = kernel.unreliableCount;
  return obj;
}

static KernelMeta parseKernelMeta(const json &value,
                                  const std::string &jsonPath,
                                  const std::string &fieldPath) {
  requireObject(value, jsonPath, fieldPath);

  KernelMeta kernel;
  kernel.kernelName =
      getString(requireField(value, jsonPath, fieldPath, "name"),
                jsonPath, fieldPath + ".name");
  const std::string kind =
      getString(requireField(value, jsonPath, fieldPath, "kind"),
                jsonPath, fieldPath + ".kind");
  kernel.kind = parseKernelKind(kind, jsonPath, fieldPath + ".kind");
  kernel.inputNames =
      getStringArray(requireField(value, jsonPath, fieldPath, "inputs"),
                     jsonPath, fieldPath + ".inputs");
  kernel.outputNames =
      getStringArray(requireField(value, jsonPath, fieldPath, "outputs"),
                     jsonPath, fieldPath + ".outputs");
  kernel.unreliableCount =
      getInt64(requireField(value, jsonPath, fieldPath, "unreliable_count"),
               jsonPath, fieldPath + ".unreliable_count");
  return kernel;
}

struct Metadata {
  std::string format;
  int64_t version = 0;
  std::string cubin;
  int64_t warpsPerCta = 1;
  int64_t vectorSize = 1;
  std::string dtype;
  std::vector<KernelMeta> kernels;
  std::vector<std::string> graphInputs;
  std::vector<std::string> graphOutputs;
  std::unordered_map<std::string, int64_t> outputUnreliable;
};

static json metadataToJSON(const ExecutableData &data,
                           const std::string &cubinName) {
  json kernels = json::array();
  for (const KernelMeta &kernel : data.kernels)
    kernels.push_back(toJSON(kernel));

  std::vector<std::string> outputNames;
  outputNames.reserve(data.outputUnreliable.size());
  for (const auto &item : data.outputUnreliable)
    outputNames.push_back(item.first);
  std::sort(outputNames.begin(), outputNames.end());

  json outputUnreliable = json::object();
  for (const std::string &name : outputNames)
    outputUnreliable[name] = data.outputUnreliable.at(name);

  json obj = json::object();
  obj["format"] = kFormat;
  obj["version"] = kVersion;
  obj["cubin"] = cubinName;
  obj["warps_per_cta"] = data.warpsPerCta;
  obj["vector_size"] = data.vectorSize;
  obj["dtype"] = toString(data.dtype);
  obj["kernels"] = std::move(kernels);
  obj["graph_inputs"] = toJSONArray(data.graphInputs);
  obj["graph_outputs"] = toJSONArray(data.graphOutputs);
  obj["output_unreliable"] = std::move(outputUnreliable);
  return obj;
}

static Metadata parseMetadata(const std::string &jsonPath,
                              const std::string &jsonText) {
  json root;
  try {
    root = json::parse(jsonText);
  } catch (const json::parse_error &e) {
    throw std::runtime_error(
        "kun_cuda::ExecutableData: failed to parse '" + jsonPath +
        "': " + e.what());
  }

  requireObject(root, jsonPath, "$");

  Metadata metadata;
  metadata.format =
      getString(requireField(root, jsonPath, "$", "format"),
                jsonPath, "$.format");
  metadata.version =
      getInt64(requireField(root, jsonPath, "$", "version"),
               jsonPath, "$.version");
  metadata.cubin =
      getString(requireField(root, jsonPath, "$", "cubin"),
                jsonPath, "$.cubin");
  metadata.warpsPerCta =
      getInt64(requireField(root, jsonPath, "$", "warps_per_cta"),
               jsonPath, "$.warps_per_cta");
  metadata.vectorSize =
      getInt64(requireField(root, jsonPath, "$", "vector_size"),
               jsonPath, "$.vector_size");
  metadata.dtype =
      getString(requireField(root, jsonPath, "$", "dtype"),
                jsonPath, "$.dtype");
  metadata.graphInputs =
      getStringArray(requireField(root, jsonPath, "$", "graph_inputs"),
                     jsonPath, "$.graph_inputs");
  metadata.graphOutputs =
      getStringArray(requireField(root, jsonPath, "$", "graph_outputs"),
                     jsonPath, "$.graph_outputs");
  metadata.outputUnreliable =
      getStringIntMap(requireField(root, jsonPath, "$", "output_unreliable"),
                      jsonPath, "$.output_unreliable");

  const json &kernelsValue = requireField(root, jsonPath, "$", "kernels");
  if (!kernelsValue.is_array())
    throw std::runtime_error(
        "kun_cuda::ExecutableData: expected array at $.kernels in '" +
        jsonPath + "'");
  metadata.kernels.reserve(kernelsValue.size());
  for (size_t i = 0; i < kernelsValue.size(); ++i)
    metadata.kernels.push_back(
        parseKernelMeta(kernelsValue[i], jsonPath,
                        "$.kernels[" + std::to_string(i) + "]"));

  return metadata;
}

} // namespace

void ExecutableData::saveToFiles(const std::string &dir,
                                 const std::string &name) const {
  validateArtifactName(name);
  ensureDirectory(dir);

  const std::string jsonName = jsonFileName(name);
  const std::string cubinName = cubinFileName(name);
  const std::string jsonPath = joinArtifactPath(dir, jsonName);
  const std::string cubinPath = joinArtifactPath(dir, cubinName);

  writeBinaryFile(cubinPath, cubin);
  writeTextFile(jsonPath, metadataToJSON(*this, cubinName).dump(2) + "\n");
}

std::shared_ptr<ExecutableData>
ExecutableData::loadFromFiles(const std::string &dir,
                              const std::string &name) {
  validateArtifactName(name);

  const std::string jsonName = jsonFileName(name);
  const std::string cubinName = cubinFileName(name);
  const std::string jsonPath = joinArtifactPath(dir, jsonName);
  const std::string cubinPath = joinArtifactPath(dir, cubinName);

  Metadata metadata = parseMetadata(jsonPath, readTextFile(jsonPath));
  if (metadata.format != kFormat)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: unsupported metadata format '" +
        metadata.format + "' in '" + jsonPath + "'");
  if (metadata.version != kVersion)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: unsupported metadata version " +
        std::to_string(metadata.version) + " in '" + jsonPath + "'");
  if (metadata.cubin != cubinName)
    throw std::runtime_error(
        "kun_cuda::ExecutableData: metadata cubin field in '" + jsonPath +
        "' must be '" + cubinName + "'");

  auto data = std::make_shared<ExecutableData>();
  data->cubin = readBinaryFile(cubinPath);
  data->warpsPerCta = metadata.warpsPerCta;
  data->vectorSize = metadata.vectorSize;
  data->dtype = parseDatatype(metadata.dtype, jsonPath);
  data->kernels = std::move(metadata.kernels);
  data->graphInputs = std::move(metadata.graphInputs);
  data->graphOutputs = std::move(metadata.graphOutputs);
  data->outputUnreliable = std::move(metadata.outputUnreliable);
  return data;
}

} // namespace kun_cuda
