import { Geist, Geist_Mono } from "next/font/google"

import "./globals.css"
import Link from "next/link"

import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/sonner"
import { cn } from "@/lib/utils";

const geist = Geist({subsets:['latin'],variable:'--font-sans'})

const fontMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn("antialiased", fontMono.variable, "font-sans", geist.variable)}
    >
      <body>
        <ThemeProvider>
          <header className="border-b">
            <div className="mx-auto flex h-14 max-w-6xl items-center px-6">
              <Link href="/" className="font-semibold">
                Insightloop
              </Link>
              <span className="ml-3 text-sm text-muted-foreground">AI research insights</span>
            </div>
          </header>
          <main className="mx-auto max-w-6xl p-6">{children}</main>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}
