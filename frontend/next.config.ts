import type { NextConfig } from "next";

const apiTarget = process.env.API_PROXY_TARGET || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${apiTarget}/api/:path*` },
      { source: "/docs", destination: `${apiTarget}/docs` },
      { source: "/openapi.json", destination: `${apiTarget}/openapi.json` },
    ];
  },
};
export default nextConfig;
