import type { Config } from "tailwindcss";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      colors: {
        bg: {
          DEFAULT: "#0b0d12",
          panel: "#11141c",
          subtle: "#161a24",
          border: "#1f2430",
          elevated: "#1a1f2e",
        },
        ink: {
          DEFAULT: "#e7eaf2",
          muted: "#9aa3b8",
          dim: "#6b7280",
        },
        accent: {
          DEFAULT: "#7c5cff",
          soft: "#7c5cff22",
        },
        success: {
          DEFAULT: "#34d399",
          soft: "#34d39922",
        },
        danger: {
          DEFAULT: "#f87171",
          soft: "#f8717122",
        },
        warn: {
          DEFAULT: "#fbbf24",
          soft: "#fbbf2422",
        },
        info: {
          DEFAULT: "#3b82f6",
          soft: "#3b82f622",
        },
      },
      boxShadow: {
        glow: "0 8px 30px -16px rgba(124, 92, 255, 0.5)",
      },
    },
  },
  plugins: [],
} satisfies Config;
