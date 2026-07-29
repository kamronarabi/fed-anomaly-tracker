"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { DonateModal } from "./DonateModal";

export function Header() {
  const [donateOpen, setDonateOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <>
      <header className="border-b border-line bg-paper">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-4">
          <Link
            href="/"
            className="flex shrink-0 items-center gap-3"
            aria-label="Fraudhound home"
          >
            <Image
              src="/fhHeaderLogo.png"
              alt="Fraudhound"
              width={300}
              height={88}
              priority
              className="h-20 w-70 shrink-0"
            />
          </Link>

          <nav className="hidden items-center gap-6 text-sm font-medium tracking-wide uppercase md:flex">
            <Link href="/" className="hover:text-brick transition-colors">
              Watchlist
            </Link>
            <Link href="/about" className="hover:text-brick transition-colors">
              About
            </Link>
            <button
              type="button"
              onClick={() => setDonateOpen(true)}
              className="rounded-full bg-brick px-5 py-2 text-paper hover:bg-brick-dark transition-colors"
            >
              ♥ Donate
            </button>
          </nav>

          <button
            type="button"
            onClick={() => setMenuOpen((open) => !open)}
            className="flex shrink-0 items-center justify-center rounded-md p-2 text-charcoal md:hidden"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {menuOpen ? (
                <>
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </>
              ) : (
                <>
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </>
              )}
            </svg>
          </button>
        </div>

        {menuOpen && (
          <nav className="flex flex-col gap-1 border-t border-line px-6 py-4 text-sm font-medium tracking-wide uppercase md:hidden">
            <Link
              href="/"
              onClick={() => setMenuOpen(false)}
              className="py-2 hover:text-brick transition-colors"
            >
              Watchlist
            </Link>
            <Link
              href="/about"
              onClick={() => setMenuOpen(false)}
              className="py-2 hover:text-brick transition-colors"
            >
              About
            </Link>
            <button
              type="button"
              onClick={() => {
                setMenuOpen(false);
                setDonateOpen(true);
              }}
              className="mt-2 w-full rounded-full bg-brick px-5 py-2 text-paper hover:bg-brick-dark transition-colors"
            >
              ♥ Donate
            </button>
          </nav>
        )}
      </header>

      <DonateModal open={donateOpen} onClose={() => setDonateOpen(false)} />
    </>
  );
}
