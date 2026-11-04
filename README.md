# Rally Cross recomp module

Rally Cross (989 Studios, 1997), statically recompiled for the TrimUI Brick. Deadstation 1.0.0 or newer hosts it when the game's Core
is set to it: Deadstation's menu, Options and Recomp Store stay Deadstation's, and this module is the game.

## Supported Formats

| Region | Format | SHA-1 |
|---|---|---|
| USA | CHD | `62c54f5e1f989126a1f034c39646a0336c6c8321` |

The SHA-1 is the CHD header's, shown by `chdman info` as **SHA1**. Other images are currently not supported.

## Features

- **Resolution:** 320 x 240, 640 x 480 or 1024 x 768, the default and the Brick's own (Options; takes effect the next time the game starts).
- **Shadows:** each car's own shape cast along the sun onto the ground, following its rises and dips (the track's heightfield), in place of the game's box under the car; each spot darkened once.
- **60 frames a second, all of it the game:** in a one-player race the physics, controls and camera run every vertical blank and every one is a whole new frame, with the game at its true speed. Decisions (the AI's, the clock's) and a wheel's spin keep the game's own rate. The physics' helpers and the ground's height are host code, to the unit what the game's give.
- **Medium High:** a camera between Medium and Far, higher and looking down the road, in the pause menu's Camera page.
- **HUD:** each part of it can be turned off, in the pause menu's HUD page.
- **The Lost Mines' tunnel:** its flames flicker orange on the rock and the cars.
- **Night races:** the AI drives at half the speed it would by day, as nobody can see far.
- **A native renderer:** a race's frame takes no PlayStation packets through the ordering table or the GPU's list: the track and the sky are meshes on the GPU, drawn whole with the camera in a shader and clipped by the GPU with a true depth, so nothing is dropped by distance. Cars, billboards, lamps and tyre tracks are drawn from host code, in the world. The HUD keeps the game's layout: its records are read as data and drawn directly. Two Player is drawn the same way, view by view, at the game's 30 (the game locks it without a second pad, and a Brick has the one).
- **Conditions:** three more pages in the game's own menus, after Direction: a season (each track's scenery recoloured), a time of day (sky, light and the length of the shadows) and the weather. Original / Daytime / Clear is the game as it shipped. In Season mode the year goes by over the races.
- **Controls:** every button can be remapped, set in Options. `module.cfg` declares them; the SDK's README explains how.

## Files

```
module.cfg              the manifest
store.json              what its page in the Recomp Store says
game.toml               the game: its disc, boot file and runtime settings (PSXRecomp)
seeds/                  the functions the translation starts from
catalog_identity.json   the disc's identity, as PSXRecomp's probe found it
pipelines.keys          the GPU pipelines it draws, built ahead of use
CMakeLists.txt          the module's build
src/plugin.cpp          the module's additions, a plugin inside the game (its hooks: game.toml)
src/conditions.h        what each season, hour and weather looks like: tables, nothing of the game's
src/shaders/            the world's shaders, with the headlights in them
mods/                   the plugin's package, which turns it on
sdk/                    the runtime and port layer
```

## Building

From your own disc:

```
git submodule update --init sdk
git -C sdk submodule update --init psxrecomp
sdk/tools/toolchain/toolchain.sh
sdk/tools/build-module.sh "/path/to/Rally Cross (USA).chd"
sdk/tools/pack-module.sh
```

## Credits

Built with [PSXRecomp](https://github.com/RetroPortingToolKit/psxrecomp).

## Licence

PolyForm Noncommercial 1.0.0 (`LICENSE`).
