/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: '/v1/:path*', destination: 'http://curation-api:8001/v1/:path*' },
      { source: '/health',    destination: 'http://curation-api:8001/health' },
    ];
  },
};

export default nextConfig;
