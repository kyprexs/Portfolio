import type React from "react"
import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "Alex West - ML Engineer",
  description: "ML Engineer portfolio showcasing NeuralScript, custom neural networks, and scientific computing projects.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>Alex West - ML Engineer</title>
        <meta name="description" content="ML Engineer portfolio showcasing NeuralScript, custom neural networks, and scientific computing projects." />
        <link rel="icon" href="/favicon-32x32.png" sizes="32x32" />
      </head>
      <body>{children}</body>
    </html>
  );
}
