import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Vite lib-mode leaves `process.env.NODE_ENV` untouched in dependencies
  // (recharts references it), but the App loads this bundle in a browser
  // where `process` does not exist, so replace it at build time.
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    lib: {
      entry: "src/index.tsx",
      name: "LeRobotDataCuration",
      fileName: "index",
      formats: ["umd"],
    },
    rollupOptions: {
      // Provided as globals by the FiftyOne App at runtime. They must not be
      // bundled, or React context and hooks split across two copies.
      external: [
        "react",
        "react-dom",
        "recoil",
        "@fiftyone/operators",
        "@fiftyone/plugins",
        "@fiftyone/state",
      ],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          recoil: "recoil",
          "@fiftyone/operators": "__foo__",
          "@fiftyone/plugins": "__fop__",
          "@fiftyone/state": "__fos__",
        },
      },
    },
  },
});
