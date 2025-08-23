import type React from "react"
import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "HT - Space Agent Portfolio",
  description: "Retro 8-bit space-themed portfolio showcasing skills, projects, and cosmic adventures.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>HT - Space Agent Portfolio</title>
        <meta name="description" content="Retro 8-bit space-themed portfolio showcasing skills, projects, and cosmic adventures." />
        <link rel="icon" href="/favicon-32x32.png" sizes="32x32" />
      </head>
      <body>{children}</body>
    </html>
  );
}
