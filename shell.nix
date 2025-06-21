let 
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-unstable") {};
in
pkgs.mkShell {
  buildInputs = with pkgs.python3Packages; [
    jax
    jaxlib
    optax
    # venvShellHook
  ] ++ (with pkgs; [
  ]);
  # venvDir = ".venv";
}
  
