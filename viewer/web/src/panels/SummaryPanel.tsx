// The run's own `context_summary.md`, rendered through `MarkdownArticle`.

import { useEffect, useState, type JSX } from "react";

import { MarkdownArticle } from "@/components/inspector/MarkdownArticle";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchTextFile } from "@/lib/api";

const SUMMARY_PATH = "context_summary.md";

type SummaryPanelProps = {
  runId: string;
};

export function SummaryPanel({ runId }: SummaryPanelProps): JSX.Element {
  const [text, setText] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setText(null);
    setTruncated(false);
    setError(null);
    fetchTextFile(runId, SUMMARY_PATH)
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
  }, [runId]);

  if (error) {
    return (
      <p className="p-3 text-[11px] text-[var(--destructive)]">
        Could not load summary: {error}
      </p>
    );
  }

  if (text == null) {
    return (
      <div className="space-y-3 p-3">
        <Skeleton className="h-5 w-2/3" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-5/6" />
        <Skeleton className="h-20 w-full" />
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="px-4 py-3">
        {truncated ? (
          <p className="mb-3 rounded-md bg-amber-500/[0.08] px-2.5 py-2 text-[10px] text-[var(--warning)]">
            This summary is truncated because it exceeds the 4 MB text limit.
          </p>
        ) : null}
        <MarkdownArticle text={text} />
      </div>
    </ScrollArea>
  );
}
