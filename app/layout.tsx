import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Dark-Store Replenishment Agent",
  description:
    "Hybrid ML + LLM-agent demand replenishment demo for quick-commerce dark stores.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
