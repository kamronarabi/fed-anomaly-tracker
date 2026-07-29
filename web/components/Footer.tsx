import Link from "next/link";

export function Footer() {
  return (
    <footer className="mt-20 border-t border-line bg-paper">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-6 py-8 text-sm text-mute sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          <Link href="/about" className="hover:text-brick transition-colors">
            About & Methodology
          </Link>
          <a
            href="mailto:kamronarabi@ufl.edu?subject=Fraudhound%20error%20report"
            className="hover:text-brick transition-colors"
          >
            Report an error
          </a>
          <a
            href="https://github.com/kamronarabi/fed-anomaly-tracker"
            target="_blank"
            rel="noreferrer"
            className="hover:text-brick transition-colors"
          >
            GitHub
          </a>
        </div>
        <div className="font-mono text-xs">
          Built independently · Data from USAspending.gov
        </div>
      </div>
    </footer>
  );
}
