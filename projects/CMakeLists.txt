
if(KUN_AVX512)
    set(AVX512ON "avx512")
else()
    set(AVX512ON "avx2")
endif()
file(GLOB PROJECTS LIST_DIRECTORIES true RELATIVE ${PROJECT_SOURCE_DIR}/projects ${PROJECT_SOURCE_DIR}/projects/*)
FOREACH(subdir ${PROJECTS})
    if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/projects/${subdir})
        file(READ ${PROJECT_SOURCE_DIR}/projects/${subdir}/list.txt raw_MODULE_DEPENDS_${subdir})
        string(REPLACE "\n" ";" tmp_MODULE_DEPENDS_${subdir} "${raw_MODULE_DEPENDS_${subdir}}")
        set(MODULE_DEPENDS_${subdir} "")
        foreach(f ${tmp_MODULE_DEPENDS_${subdir}})
            list(APPEND MODULE_DEPENDS_${subdir} ${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${f})
        endforeach()
        
        # message("EXEC ${MODULE_DEPENDS_${subdir}}")
        add_custom_command(
            OUTPUT ${MODULE_DEPENDS_${subdir}}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/projects/${subdir}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/${subdir} ${AVX512ON}
            DEPENDS ${PROJECT_SOURCE_DIR}/projects/${subdir}/generate.py ${PROJECT_SOURCE_DIR}/projects/${subdir}/list.txt
        )
        add_library(${subdir} SHARED ${MODULE_DEPENDS_${subdir}})
        target_link_libraries(${subdir} KunRuntime)
        message(STATUS "Added project: ${subdir}, src: ${tmp_MODULE_DEPENDS_${subdir}}")
    endif()
ENDFOREACH()