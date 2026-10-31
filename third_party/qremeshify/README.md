# QRemeshify (QuadWild + Bi-MDF)

LAMP calls QuadWild (Pietroni et al. 2021) with Bi-MDF quantisation (Heistermann et al. 2023) through the prebuilt
libraries shipped with **QRemeshify v1.1.0** (<https://github.com/ksami/QRemeshify>, GPL-3.0-or-later).
The binaries are not redistributed here.

1. Download the QRemeshify **v1.1.0** release archive for your platform (windows-x64, linux-x64 or macos-arm64)
   from <https://github.com/ksami/QRemeshify/releases>.
2. Extract it so that this path exists: `third_party/qremeshify/QRemeshify/lib/__init__.py`
   (next to `lib_quadwild` / `lib_quadpatches` shared libraries),
   **or** point the environment variable `QREMESHIFY_DIR` at the extracted `QRemeshify` folder.
3. Check: `python -c "import sys; sys.path.insert(0, 'src'); from lamp_mesh.quadwild import _lib; print(_lib())"`

Only the `lib/` part of the add-on is used (ctypes bindings); Blender is not needed.
All reported numbers were produced with the Windows x64 build.
