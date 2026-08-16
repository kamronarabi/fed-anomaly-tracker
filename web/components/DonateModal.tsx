"use client";

import { useEffect, useState } from "react";

type Preset = 5 | 25 | 100;

interface DonateModalProps {
  open: boolean;
  onClose: () => void;
}

export function DonateModal({ open, onClose }: DonateModalProps) {
  const [selected, setSelected] = useState<Preset | "custom">(25);
  const [custom, setCustom] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Close on ESC.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  async function submit() {
    setError(null);
    let dollars: number;
    if (selected === "custom") {
      dollars = Number(custom);
      if (!Number.isFinite(dollars) || dollars < 1) {
        setError("Enter an amount of $1 or more.");
        return;
      }
    } else {
      dollars = selected;
    }
    const amount_cents = Math.round(dollars * 100);
    if (amount_cents > 100000000000) {
      setError("Custom donations capped at $1,000 in v1.");
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch("/api/stripe/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ amount_cents }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Server returned ${res.status}`);
      }
      const { url } = await res.json();
      if (!url) throw new Error("No checkout URL returned.");
      window.location.href = url;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setSubmitting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-charcoal/60 px-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="donate-modal-title"
    >
      <div
        className="relative w-full max-w-md rounded-lg bg-paper p-8 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="absolute right-4 top-4 text-mute hover:text-charcoal text-xl leading-none"
        >
          ✕
        </button>

        <h2
          id="donate-modal-title"
          className="font-serif text-2xl font-bold tracking-tight"
        >
          Support Fraudhound
        </h2>
        <p className="mt-3 text-sm text-charcoal/80 leading-relaxed">
          Independent statistical scrutiny of federal contracting. No ads,
          no paywalls, no funders to answer to. Your donation covers
          servers, AI costs, and the time to keep
          chasing the data.
        </p>

        <div className="mt-6 flex flex-wrap gap-2">
          {([5, 25, 100] as Preset[]).map((amt) => (
            <button
              key={amt}
              type="button"
              onClick={() => {
                setSelected(amt);
                setError(null);
              }}
              className={`rounded-md border px-4 py-2 font-medium transition-colors ${
                selected === amt
                  ? "border-brick bg-brick text-paper"
                  : "border-line bg-paper text-charcoal hover:border-brick"
              }`}
            >
              ${amt}
            </button>
          ))}
          <button
            type="button"
            onClick={() => {
              setSelected("custom");
              setError(null);
            }}
            className={`rounded-md border px-4 py-2 font-medium transition-colors ${
              selected === "custom"
                ? "border-brick bg-brick text-paper"
                : "border-line bg-paper text-charcoal hover:border-brick"
            }`}
          >
            Custom
          </button>
        </div>

        {selected === "custom" && (
          <div className="mt-3 flex items-center gap-2">
            <span className="text-mute">$</span>
            <input
              type="number"
              min={1}
              max={1000}
              step={1}
              value={custom}
              onChange={(e) => setCustom(e.target.value)}
              placeholder="50"
              className="w-32 rounded-md border border-line bg-paper px-3 py-2 outline-none focus:border-brick"
            />
          </div>
        )}

        {error && <p className="mt-4 text-sm text-brick">{error}</p>}

        <button
          type="button"
          onClick={submit}
          disabled={submitting}
          className="mt-6 w-full rounded-md bg-charcoal py-3 font-medium text-paper hover:bg-charcoal/90 transition-colors disabled:opacity-60"
        >
          {submitting ? "Redirecting…" : "Donate"}
        </button>

        <p className="mt-4 text-xs text-mute">
          Powered by Stripe Checkout 
        </p>
      </div>
    </div>
  );
}
