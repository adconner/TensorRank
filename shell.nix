let pkgs = import <nixpkgs> {}; 
in
pkgs.mkShell {
  buildInputs = with pkgs.python3Packages; [
    ipython
    jax
    jaxlib
    jax-cuda12-plugin
    optax
    optimistix
  ];
}
  
