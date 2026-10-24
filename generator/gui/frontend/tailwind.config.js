/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#FAFAF9",
        surface: "#FFFFFF",
        surfaceElevated: "#F5F5F4",
        surfaceHover: "#EBEBEA",
        border: "#E7E5E4",
        borderSubtle: "#F0EFEB",
        copper: {
          50: "#FFF7ED",
          100: "#FFEDD5",
          200: "#FED7AA",
          300: "#FDBA74",
          400: "#FB923C",
          500: "#C86432",
          600: "#B85526",
          700: "#9C4118",
          800: "#7C3214",
          900: "#5E250E",
        },
        accent: "#B85526",
        accentHover: "#9C4118",
      },
    },
  },
  plugins: [],
}
