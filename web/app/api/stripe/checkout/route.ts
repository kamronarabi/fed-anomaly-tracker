// Stripe Checkout session creator. Called from the Donate modal.
//
// Body: { amount_cents: number }
// Response: { url: string }  // Stripe-hosted checkout URL to redirect to.

import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";

const MIN_CENTS = 100;        // $1
const MAX_CENTS = 100_000;    // $1,000 cap in v1

function getStripe(): Stripe {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) {
    throw new Error("STRIPE_SECRET_KEY not set");
  }
  return new Stripe(key);
}

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const amount_cents = Number(
    (body as { amount_cents?: unknown })?.amount_cents,
  );
  if (!Number.isInteger(amount_cents)) {
    return NextResponse.json(
      { error: "amount_cents must be an integer" },
      { status: 400 },
    );
  }
  if (amount_cents < MIN_CENTS || amount_cents > MAX_CENTS) {
    return NextResponse.json(
      {
        error: `amount_cents must be between ${MIN_CENTS} ($1) and ${MAX_CENTS} ($1,000)`,
      },
      { status: 400 },
    );
  }

  const origin =
    req.headers.get("origin") ??
    process.env.NEXT_PUBLIC_SITE_URL ??
    "http://localhost:3000";

  let session;
  try {
    const stripe = getStripe();
    session = await stripe.checkout.sessions.create({
      mode: "payment",
      payment_method_types: ["card"],
      line_items: [
        {
          price_data: {
            currency: "usd",
            unit_amount: amount_cents,
            product_data: {
              name: "Support Fraudhound",
              description:
                "Independent statistical scrutiny of federal contracting.",
            },
          },
          quantity: 1,
        },
      ],
      success_url: `${origin}/donate/thanks?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${origin}/`,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Stripe error";
    console.error("checkout error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }

  if (!session.url) {
    return NextResponse.json(
      { error: "Stripe session did not return a URL" },
      { status: 500 },
    );
  }

  return NextResponse.json({ url: session.url });
}
