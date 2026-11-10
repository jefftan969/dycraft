// Any text artifact in the run dir: the agent's `object.py` / `solids.py`, the
// generated URDF, a part's staged build123d source, `ir.json`.
//
// Articraft's CodePanel is not vendored — it is bound to their record/staging
// fetchers and a fixed `model.py` / `model.urdf` tab pair. This keeps their
// syntax-highlighting module and drives it off the run's actual file list.

import { useEffect, useState, type JSX } from "react";

import {
  codeTheme,
  normalizeHighlightLanguage,
  SyntaxHighlighter,
} from "@/components/inspector/syntax-highlighting";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchTextFile } from "@/lib/api";
import { cn } from "@/lib/utils";

const LANGUAGE_BY_SUFFIX: Record<string, string> = {
  py: "python",
  json: "json",
  jsonl: "json",
  urdf: "xml",
  xml: "xml",
  md: "markdown",
};

function languageFor(path: string): "json" | "python" | "xml" | "text" {
  const suffix = path.split(".").pop()?.toLowerCase() ?? "";
  // The highlighter only ships three grammars; anything else renders as plain.
  return normalizeHighlightLanguage(LANGUAGE_BY_SUFFIX[suffix]) ?? "text";
}

type CodePanelProps = {
  runId: string;
  files: string[];
  selected: string | null;
  onSelect: (path: string) => void;
};

export function CodePanel({
  runId,
  files,
  selected,
  onSelect,
}: CodePanelProps): JSX.Element {
  const [text, setText] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const path = selected ?? files[0] ?? null;

  useEffect(() => {
    if (!path) {
      setText(null);
      return;
    }
    let cancelled = false;
    setText(null);
    setError(null);
    fetchTextFile(runId, path)
      .then((file) => {
        if (cancelled) return;
        setText(file.text);
        setTruncated(file.truncated);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, path]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap gap-1 border-b border-[var(--border-subtle)] p-2">
        {files.map((file) => (
          <button
            key={file}
            type="button"
            onClick={() => onSelect(file)}
            className={cn(
              "rounded px-1.5 py-0.5 font-mono text-[10px]",
              file === path
                ? "bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
                : "text-[var(--text-tertiary)] hover:text-[var(--text-primary)]",
            )}
          >
            {file}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1">
        {error ? (
          <p className="p-3 text-[11px] text-red-400">{error}</p>
        ) : text == null ? (
          <div className="space-y-2 p-3">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        ) : (
          <ScrollArea className="h-full">
            {truncated ? (
              <p className="px-3 pt-2 text-[10px] text-amber-500">
                Truncated — file exceeds the 4 MB text limit.
              </p>
            ) : null}
            <SyntaxHighlighter
              language={languageFor(path ?? "")}
              style={codeTheme}
              customStyle={{
                background: "transparent",
                fontSize: "11px",
                margin: 0,
                padding: "12px",
              }}
              showLineNumbers
            >
              {text}
            </SyntaxHighlighter>
          </ScrollArea>
        )}
      </div>
    </div>
  );
}
