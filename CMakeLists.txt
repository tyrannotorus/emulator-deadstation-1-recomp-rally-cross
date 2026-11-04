cmake_minimum_required(VERSION 3.20)
# The Rally Cross recomp module: Rally Cross (US, SCUS-94308) statically recompiled with PSXRecomp
# (sdk/psxrecomp, PolyForm Noncommercial 1.0.0), from the builder's own disc. Its game C (generated/),
# disc staging (disc/) and overlay captures (overlays/) come from that disc and never go in git.
set(PSXRECOMP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sdk/psxrecomp" CACHE PATH "Path to psxrecomp")
if(NOT EXISTS "${PSXRECOMP_ROOT}/runtime/runtime.cmake")
    message(FATAL_ERROR "PSXRecomp runtime is missing: ${PSXRECOMP_ROOT}/runtime/runtime.cmake "
        "(git submodule update --init sdk; git -C sdk submodule update --init psxrecomp; sdk/tools/prepare-psxrecomp.sh)")
endif()

project(RallyCrossRecomp LANGUAGES C CXX)

set(PSX_RECOMP_UI OFF CACHE BOOL "No recomp-ui" FORCE)
include("${PSXRECOMP_ROOT}/runtime/runtime.cmake")

set(PSX_PRELOADED_MODS "${CMAKE_CURRENT_SOURCE_DIR}/mods/preloaded")
if(NOT IS_DIRECTORY "${PSX_PRELOADED_MODS}")
    set(PSX_PRELOADED_MODS "NONE")
endif()

psxrecomp_add_game_runtime(psx-runtime
    PRELOADED_MODS_DIR "${PSX_PRELOADED_MODS}"
    WINDOW_TITLE "Rally Cross recomp module"
    DEFAULT_GAME_CONFIG_PATH "game.toml"
    MAX_PLAYERS 4
    # captured overlays baked in natively (psxrecomp/tools/compile_overlays.py --static); optional until the
    # first bake; must precede GEN_FULL_GLOB
    GAME_OVERLAY_STATIC_C "${CMAKE_CURRENT_SOURCE_DIR}/generated/overlays_static.c"
    APP_ICON "${PSXRECOMP_ROOT}/assets/psxrecomp.ico"
    GEN_MARKER "generated/SCUS_943.08_dispatch.c"
    GEN_FULL_GLOB "generated/SCUS_943.08_full_*.c"
)

# the module's plugin in the game (its hooks: game.toml mod_function_entry_funcs)
target_sources(psx-runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/sdk/brick/psx_sdk.cpp")

# the Brick: the presenter, no desktop OpenGL (sdk/brick/brick.cmake)
include("${CMAKE_CURRENT_SOURCE_DIR}/sdk/brick/brick.cmake")

# the headlights: this game's own world shaders (src/shaders)
if(COMMAND deadstation_module_shaders)
    deadstation_module_shaders("${CMAKE_CURRENT_SOURCE_DIR}/src/shaders" world.vert world_lit.vert world_lit.frag world_edge_lit.frag)
endif()

foreach(_PSX_RUNTIME_CONFIG game.toml game_options.toml)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_PSX_RUNTIME_CONFIG}")
        add_custom_command(TARGET psx-runtime POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/${_PSX_RUNTIME_CONFIG}"
                "$<TARGET_FILE_DIR:psx-runtime>/${_PSX_RUNTIME_CONFIG}"
            VERBATIM)
    endif()
endforeach()
