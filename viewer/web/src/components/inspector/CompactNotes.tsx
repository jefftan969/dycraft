// A list of notes as one line under the thing they are about, expandable: the first note, "+N more", and a label
// and tooltip saying what they are for.
//
// Two kinds, told apart by color: standing notes, about how something is shown or derived (gray), and problems
// (amber, and listed first). A lone standing note is shown as it is, with nothing to expand.

import { useState, type JSX } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

export function CompactNotes({
  label,
  title,
  standing,
  problems = [],
  className,
}: {
  /** Shown before the first note: what the notes are about. */
  label: string;
  /** The tooltip: what the notes are for. */
  title: string;
  standing: string[];
  problems?: string[];
  className?: string;
}): JSX.Element | null {
  const [open, setOpen] = useState(false);
  const notes = [...problems.map((text) => ({ text, problem: true })),
                 ...standing.map((text) => ({ text, problem: false }))];
  if (notes.length === 0) return null;
  const frame = cn("border-t border-[var(--border-subtle)] bg-[var(--surface-1)] px-3 py-1 text-[10px]", className);
  if (problems.length === 0 && standing.length === 1) {
    return (
      <div className={frame}>
        <p className="truncate text-[var(--text-tertiary)]" title={standing[0]}>{standing[0]}</p>
      </div>
    );
  }
  const tone = problems.length > 0 ? "text-[var(--warning)]" : "text-[var(--text-tertiary)]";
  return (
    <div className={frame}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        title={title}
        className={cn("flex w-full min-w-0 items-center gap-1.5 text-left", tone)}
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span className="shrink-0 font-medium">{label}:</span>
        <span className="min-w-0 truncate">{notes[0].text}</span>
        {notes.length > 1 ? <span className="shrink-0 opacity-75">+{notes.length - 1} more</span> : null}
      </button>
      {open ? (
        <ul className="mt-1 max-h-32 space-y-0.5 overflow-y-auto pl-[18px]">
          {notes.map((note) => (
            <li key={note.text} className={note.problem ? "text-[var(--warning)]" : "text-[var(--text-tertiary)]"}>
              {note.text}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
