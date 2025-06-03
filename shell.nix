let 
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-25.05") {};
# let pkgs = import <nixpkgs> {}; 
in
pkgs.mkShell {
  buildInputs = with pkgs.python3Packages; [
    # tensorflow
    # ipython
    jax
    jaxlib
    jax-cuda12-plugin
    optax
    # optimistix
    venvShellHook
  ];
  venvDir = ".venv";
}
  
