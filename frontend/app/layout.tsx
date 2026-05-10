/**
 * Root layout for the UX LLM Testing application.
 *
 * Wraps all pages with a shared Navbar and sets the HTML language to Slovak.
 * This layout is rendered once and persists across client-side navigation.
 */
import "./globals.css";
import Navbar from "./components/navbar";

export const metadata = {
  title: "Bakalarsky projekt",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="sk">
      <body>
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}