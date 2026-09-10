import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'
import svgr from 'vite-plugin-svgr'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd()) as {
    VITE_API_BASE: string
    VITE_API_PROXY: string
  }

  return {
    server: {
      port: 5181,
      host: '0.0.0.0',

      proxy: {
        [env.VITE_API_BASE]: {
          target: env.VITE_API_PROXY,
          changeOrigin: true,
          secure: false,
          // The backend routes do not include the frontend's proxy prefix.
          rewrite: (path) =>
            path.replace(new RegExp(`^${env.VITE_API_BASE}`), ''),
        },
      },
    },
    resolve: {
      alias: [
        {
          find: /^@\//,
          replacement: '/src/',
        },
        {
          // The graph viewer uses WebGL. Exclude the renderer package's
          // optional WebGPU backend so it is not shipped to every graph route.
          find: /^three\/webgpu$/,
          replacement: '/src/components/graph-viewer/webgpu-disabled.ts',
        },
      ],
    },

    build: {
      rollupOptions: {
        output: {
          codeSplitting: {
            includeDependenciesRecursively: false,
            maxSize: 450_000,
            groups: [
              {
                name: 'vendor-three-core',
                test: /node_modules[\\/]three[\\/]build[\\/]three\.core\.js$/,
              },
              {
                name: 'vendor-three-main',
                test: /node_modules[\\/]three[\\/]build[\\/]three\.module\.js$/,
              },
              {
                name: 'vendor-three-extras',
                test: /node_modules[\\/]three[\\/]/,
              },
              {
                name: 'vendor-graph-rendering',
                test: /node_modules[\\/](?:react-force-graph-3d|3d-force-graph|three-forcegraph|three-render-objects|three-spritetext|float-tooltip)[\\/]/,
              },
              {
                name: 'vendor-graph-math',
                test: /node_modules[\\/](?:d3-(?:array|binarytree|color|dispatch|drag|ease|force-3d|interpolate|octree|quadtree|scale|scale-chromatic|selection|timer|transition|zoom)|ngraph\.(?:events|forcelayout|graph|merge|random)|bezier-js|polished|tinycolor2|@tweenjs[\\/]tween\.js)[\\/]/,
              },
              {
                name: 'vendor-graph-runtime',
                test: /node_modules[\\/](?:accessor-fn|canvas-color-tracker|data-bind-mapper|index-array-by|jerrypick|kapsule|react-kapsule)[\\/]/,
              },
              {
                name: 'vendor-react',
                test: /node_modules[\\/](?:@remix-run[\\/]router|react|react-dom|react-router|react-router-dom|scheduler)[\\/]/,
              },
              {
                name: 'vendor-data',
                test: /node_modules[\\/](?:axios|lodash-es|proxy-compare|valtio)[\\/]/,
              },
              {
                name: 'vendor-ui-runtime',
                test: /node_modules[\\/](?:@ant-design[\\/](?:cssinjs|cssinjs-utils|fast-color)|@emotion[\\/]hash|rc-util|resize-observer-polyfill|stylis)[\\/]/,
              },
              {
                name: 'vendor-form',
                test: /node_modules[\\/](?:@rc-component[\\/]async-validator|rc-field-form)[\\/]/,
              },
            ],
          },
        },
      },
    },

    plugins: [react(), svgr()],
  }
})
