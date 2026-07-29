import type { Metadata } from "next";
import { ALL_DETECTORS } from "@/lib/types";
import { DETECTOR_META } from "@/lib/detector-meta";

export const metadata: Metadata = {
  title: "About — Fraudhound",
  description:
    "How Fraudhound surfaces statistical anomalies in federal contracting: detectors, briefs, data sources, and caveats.",
};

export default function AboutPage() {
  return (
    <article className="mx-auto max-w-3xl px-6 py-12 font-serif">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
        About Fraudhound
      </h1>

      <div className="mt-8 space-y-5 text-lg leading-relaxed">
        <p>
          Fraudhound is an independent watchdog that uses statistical
          anomaly detection to scrutinize federal contracting. Every
          night, we score every contractor in the federal contracting
          database against six statistical detectors. The 50 most
          anomalous get profiled in a daily briefing.
        </p>
        <p>
          We are not the FBI. We do not allege fraud. We surface
          statistical patterns that <em>can</em> indicate fraud, scope
          creep, non-competitive procurement, or contract gaming — and we
          show our work so you can judge for yourself.
        </p>
      </div>

      <Section title="How we detect" anchor="how-we-detect">
        <p className="mb-6 text-base leading-relaxed">
          Six statistical tests run every night against the full federal
          contracting database.
        </p>
        <div className="space-y-8">
          {ALL_DETECTORS.map((name) => {
            const meta = DETECTOR_META[name];
            return (
              <div key={name} id={meta.anchorId}>
                <h3 className="font-sans text-sm font-semibold uppercase tracking-[0.18em] text-brick">
                  {meta.longName}
                </h3>
                <p className="mt-3 text-base leading-relaxed">
                  {meta.shortDescription}
                </p>
              </div>
            );
          })}
        </div>
      </Section>

      <Section title="The briefs" anchor="briefs">
        <p className="text-base leading-relaxed">
          For the top 50 contractors each day, we generate a short
          three-paragraph brief using Claude. The model receives only the
          structured detector outputs, so the same inputs always produce the same brief.
          Every brief is auditable and reproducible from public data.
        </p>
      </Section>

      <Section title="Data sources" anchor="data">
        <dl className="grid grid-cols-1 gap-3 text-base sm:grid-cols-[160px_1fr]">
          <dt className="font-sans uppercase tracking-wide text-mute text-xs sm:pt-1">
            Contract data
          </dt>
          <dd>USAspending.gov (public API + archives)</dd>
          <dt className="font-sans uppercase tracking-wide text-mute text-xs sm:pt-1">
            Coverage
          </dt>
          <dd>Department of Defense + Health and Human Services, FY2024–2026</dd>
          <dt className="font-sans uppercase tracking-wide text-mute text-xs sm:pt-1">
            Refresh
          </dt>
          <dd>Nightly, ~2am ET</dd>
        </dl>
      </Section>

      <Section title="What we don't claim" anchor="caveats">
        <ul className="space-y-2 text-base leading-relaxed">
          <li>
            <span className="text-brick mr-2">·</span>A high composite
            score is not proof of fraud.
          </li>
          <li>
            <span className="text-brick mr-2">·</span>Detectors are
            statistical signals, not legal evidence.
          </li>
          <li>
            <span className="text-brick mr-2">·</span>Many flagged
            contractors are large established firms whose anomalies have
            legitimate explanations.
          </li>
          <li>
            <span className="text-brick mr-2">·</span>Investigation,
            audit, or document review is required before drawing
            conclusions.
          </li>
        </ul>
      </Section>

      <Section title="Who built this" anchor="who">
        <p className="text-base leading-relaxed">
          Fraudhound was built by Kamron Arabi as an independent
          accountability project. Source code is open on{" "}
          <a
            href="https://github.com/kamronarabi/fed-anomaly-tracker"
            target="_blank"
            rel="noreferrer"
            className="text-brick underline-offset-2 hover:underline"
          >
            GitHub
          </a>
          .
        </p>
        <p className="mt-3 text-base leading-relaxed">
          Contact:{" "}
          <a
            href="mailto:kamronarabi@ufl.edu"
            className="text-brick underline-offset-2 hover:underline"
          >
            kamronarabi@ufl.edu
          </a>
        </p>
      </Section>
    </article>
  );
}

function Section({
  title,
  anchor,
  children,
}: {
  title: string;
  anchor: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-12 border-t border-line pt-8" id={anchor}>
      <h2 className="font-serif text-2xl font-bold tracking-tight">{title}</h2>
      <div className="mt-5 text-charcoal/90">{children}</div>
    </section>
  );
}
