import type React from "react"
import type { Metadata } from "next"
import "./globals.css"
import { Inter } from "next/font/google"

export const metadata: Metadata = {
  title: "Alexander West – Portfolio",
  description: "Personal portfolio site for Alexander West showcasing full-stack projects, core skills, and contact info.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>Alexander West – Portfolio</title>
        <meta name="description" content="Personal portfolio site for Alexander West showcasing full-stack projects, core skills, and contact info." />
        <link rel="icon" href="/favicon-32x32.png" sizes="32x32" />
      </head>
      <body>{children}</body>
    </html>
  );
}
