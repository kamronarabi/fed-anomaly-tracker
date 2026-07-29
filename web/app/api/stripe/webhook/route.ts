// Stripe webhook receiver. Verifies the signature using STRIPE_WEBHOOK_SECRET
// and logs payment confirmations. v1 doesn't persist donations — we just log
// so Railway/Vercel captures the event in case we need to debug.

import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";

// Raw body is required to verify Stripe's signature; the App Router gives
// us the raw text via `req.text()`.
export async function POST(req: NextRequest) {
  const secret = process.env.STRIPE_SECRET_KEY;
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!secret || !webhookSecret) {
    console.error("Stripe env vars missing");
    return NextResponse.json({ error: "Server misconfigured" }, { status: 500 });
  }

  const signature = req.headers.get("stripe-signature");
  if (!signature) {
    return NextResponse.json({ error: "Missing signature" }, { status: 400 });
  }

  const stripe = new Stripe(secret);
  const rawBody = await req.text();

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(rawBody, signature, webhookSecret);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Bad signature";
    console.error("webhook signature error:", message);
    return NextResponse.json({ error: message }, { status: 400 });
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    console.log(
      "fraudhound: donation received",
      JSON.stringify({
        session_id: session.id,
        amount_total: session.amount_total,
        currency: session.currency,
        payment_status: session.payment_status,
      }),
    );
  }

  return NextResponse.json({ received: true });
}
