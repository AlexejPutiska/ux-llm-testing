/**
 * Navbar component — persistent top navigation bar.
 *
 * Renders links to all main pages of the application and highlights
 * the currently active route using the `active` CSS class.
 */
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Navigation links rendered in the header
const links = [
  { href: "/o_projekte", label: "O projekte" },
  { href: "/sprava_testov", label: "Správa testov" },
  { href: "/sprava_person", label: "Správa person" },
  { href: "/historia_testov", label: "História testovania" },
  { href: "/spustit_test", label: "Spustiť testovanie" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="navbar">
      <nav className="navbar-inner" aria-label="Hlavná navigácia">
        {links.map((l) => {
          const active = pathname === l.href;

          return (
            <Link
              key={l.href}
              href={l.href}
              className={`navbar-link ${active ? "active" : ""}`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}