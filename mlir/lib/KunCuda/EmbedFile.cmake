# Embed a binary file as a C++ unsigned-char array.
#
# Usage: cmake -DINPUT=foo.ptx -DOUTPUT=foo_ptx.inc -DSYMBOL=kun_cs_rank_ptx
#              [-DPTX_VERSION=7.8]   # optional: rewrite the `.version` directive
#              -P EmbedFile.cmake
#
# Produces (in OUTPUT):
#   static const unsigned char SYMBOL[] = { 0x12, 0x34, ... };
#   static const unsigned int  SYMBOL_len = N;
#
# If PTX_VERSION is set, the input is read as text and its first
# `.version X.Y` line is replaced before encoding — useful when nvcc
# emits a newer ISA than the deployed CUDA driver supports.

if(NOT INPUT OR NOT OUTPUT OR NOT SYMBOL)
  message(FATAL_ERROR
      "EmbedFile.cmake: INPUT, OUTPUT and SYMBOL must all be set")
endif()

if(PTX_VERSION)
  file(READ "${INPUT}" text_content)
  string(REGEX REPLACE "\\.version[ \\t]+[0-9.]+" ".version ${PTX_VERSION}"
                       text_content "${text_content}")
  set(_patched "${OUTPUT}.raw.ptx")
  file(WRITE "${_patched}" "${text_content}")
  file(READ "${_patched}" hex_content HEX)
  file(REMOVE "${_patched}")
else()
  file(READ "${INPUT}" hex_content HEX)
endif()
string(LENGTH "${hex_content}" hex_len)
math(EXPR n_bytes "${hex_len} / 2")

# "abcd" → "0xab,0xcd,"
string(REGEX REPLACE "(..)" "0x\\1," byte_list "${hex_content}")
# Trim the trailing comma.
string(REGEX REPLACE ",$" "" byte_list "${byte_list}")
# Insert a newline every 16 bytes to keep the generated file diffable.
string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)"
                     "\\1\n  " byte_list "${byte_list}")

file(WRITE "${OUTPUT}"
"// Generated from \"${INPUT}\".  Do not edit by hand.
static const unsigned char ${SYMBOL}[] = {
  ${byte_list}
};
static const unsigned int ${SYMBOL}_len = ${n_bytes};
")
