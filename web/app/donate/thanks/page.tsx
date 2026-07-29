import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Thank you — Fraudhound",
};

export default function DonateThanksPage() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-20 text-center">
      <h1 className="font-serif text-4xl font-bold tracking-tight sm:text-5xl">
        Thank you.
      </h1>
      <p className="mt-6 font-serif text-lg leading-relaxed text-charcoal/85">
        Your donation supports an independent statistical scrutiny of
        federal contracting. No ads, no paywalls, no funders to answer to.
      </p>
      <p className="mt-3 font-serif italic text-mute">
        — Kamron
      </p>
      <Link
        href="/"
        className="mt-10 inline-block rounded-full bg-brick px-6 py-3 font-medium text-paper hover:bg-brick-dark transition-colors"
      >
        Back to the watchlist
      </Link>
    </div>
  );
}
