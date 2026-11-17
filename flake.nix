{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane = {
      url = "github:ipetkov/crane";
    };
    # advisory-db = {
    #   url = "github:rustsec/advisory-db";
    #   flake = false;
    # };
    treefmtSrc.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      crane,
      rust-overlay,
      # advisory-db,
      treefmtSrc,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs =
          (import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ rust-overlay.overlays.default ];
          }).extend
            (
              self: super: {
                rust_stable = self.rust-bin.stable.latest.default;
                rust_nightly = self.rust-bin.nightly.latest.default;
              }
            );

        treefmt = treefmtSrc.lib.evalModule pkgs ./nix/treefmt.nix;

        llvm = pkgs.llvmPackages_22;
        stdenv = llvm.stdenv;
        mkShell = pkgs.mkShell.override { inherit stdenv; };
        crane' = (crane.mkLib pkgs).overrideToolchain (pkgs.rust_stable);

        sourceFilter =
          path: type:
          (crane'.filterCargoSources path type)
          || (pkgs.lib.hasSuffix ".proto" path)
          || (pkgs.lib.hasSuffix "config.json" path)
          || (pkgs.lib.hasSuffix ".onnx" path)
          || (pkgs.lib.hasSuffix ".h" path)
          || (pkgs.lib.hasSuffix ".tiktoken" path)
          || (pkgs.lib.hasInfix "/src/test/fixtures/" path);

        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = sourceFilter;
        };

        onnxTestData = pkgs.fetchFromGitHub {
          owner = "onnx";
          repo = "onnx";
          rev = "bd577f8df5b3fc58a171471125fbda1f7486b5e8";
          hash = "sha256-UclqX+WcrU2ZGLPoH+7ZHABM8Jqzc4BmHC8UGhyF/3k=";
          sparseCheckout = [
            "onnx/backend/test/data/node"
            "onnx/backend/test/data/light"
          ];
        };

        nativeBuildInputs = with pkgs; [
          pkgconf
          openssl.dev
          protobuf
          libffi
          libxml2
          z3
          zlib
          llvm.clang
          # llvm-objdump and llvm-dwarfdump, which the SQTT tests read code objects with.
          (pkgs.lib.getBin llvm.llvm)
        ];

        commonArgs = {
          inherit src nativeBuildInputs;
          LIBCLANG_PATH = "${llvm.libclang.lib}/lib/";
          ONNX_TEST_DATA = "${onnxTestData}/onnx/backend/test/data";
          # cc-wrapper appends NIX_HARDENING_ENABLE flags to *every* clang call,
          # including the runtime `clang --target=amdgcn-amd-amdhsa` cross-compile
          # in the AMD backend tests. The AMDGPU target rejects host-oriented
          # flags, so drop the two that break the build:
          #   - fortify: _FORTIFY_SOURCE needs optimization; debug builds are -O0.
          #   - zerocallusedregs: `-fzero-call-used-regs=used-gpr` is unsupported
          #     for amdgcn and fails the compile-smoke test.
          hardeningDisable = [
            "fortify"
            "zerocallusedregs"
          ];

        };

        cargoArtifacts = crane'.buildDepsOnly (commonArgs // { });
      in
      {
        checks = {
          clippy = crane'.cargoClippy (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );

          test = crane'.cargoNextest (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoNextestExtraArgs = "--features z3,proptest -E 'not test(light_densenet121)'";
            }
          );

          # UC2: the JIT ELF loader path must keep working when the crate is
          # built with `dlopen-fallback`; the default check never exercises it.
          test-dlopen-fallback = crane'.cargoNextest (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoNextestExtraArgs = "-p svod-runtime --features dlopen-fallback";
            }
          );

          # audit = crane'.cargoAudit {
          # inherit src advisory-db;
          # };

          rustfmt = crane'.cargoFmt { inherit src; };
          # treefmt = treefmt.config.build.check self;
        };

        devShells = rec {
          stable = mkShell (
            commonArgs
            // {
              packages =
                (with pkgs; [
                  rust_stable
                  cargo-outdated
                  git
                ])
                ++ nativeBuildInputs;
            }
          );

          nightly = mkShell (
            commonArgs
            // {
              packages =
                (with pkgs; [
                  rust_stable
                  cargo-outdated
                  git
                ])
                ++ nativeBuildInputs
                ++ [ pkgs.cargo-udeps ];
            }
          );

          default = stable;
        };

        formatter = treefmt.config.build.wrapper;
      }
    );
}
