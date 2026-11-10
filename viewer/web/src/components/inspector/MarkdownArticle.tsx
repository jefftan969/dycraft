// Markdown rendered in the inspector's own type scale, used by `SummaryPanel`.

import type { ComponentPropsWithoutRef, JSX } from "react";
import Markdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

const COMPONENTS: Components = {
  h1: ({ children }: ComponentPropsWithoutRef<"h1">) => (
    <h1 className="mb-3 mt-5 text-[17px] font-semibold leading-tight tracking-[-0.02em] text-[var(--text-primary)] first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }: ComponentPropsWithoutRef<"h2">) => (
    <h2 className="mb-2 mt-5 border-b border-[var(--border-subtle)] pb-1.5 text-[13px] font-semibold leading-snug text-[var(--text-primary)] first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }: ComponentPropsWithoutRef<"h3">) => (
    <h3 className="mb-1.5 mt-4 text-[12px] font-semibold text-[var(--text-primary)] first:mt-0">
      {children}
    </h3>
  ),
  h4: ({ children }: ComponentPropsWithoutRef<"h4">) => (
    <h4 className="mb-1 mt-3 text-[11px] font-semibold text-[var(--text-primary)]">
      {children}
    </h4>
  ),
  h5: ({ children }: ComponentPropsWithoutRef<"h5">) => (
    <h5 className="mb-1 mt-3 text-[11px] font-medium text-[var(--text-primary)]">
      {children}
    </h5>
  ),
  h6: ({ children }: ComponentPropsWithoutRef<"h6">) => (
    <h6 className="mb-1 mt-3 font-mono text-[10px] font-medium uppercase tracking-wide text-[var(--text-tertiary)]">
      {children}
    </h6>
  ),
  p: ({ children }: ComponentPropsWithoutRef<"p">) => (
    <p className="my-2 first:mt-0 last:mb-0">{children}</p>
  ),
  strong: ({ children }: ComponentPropsWithoutRef<"strong">) => (
    <strong className="font-semibold text-[var(--text-primary)]">
      {children}
    </strong>
  ),
  em: ({ children }: ComponentPropsWithoutRef<"em">) => (
    <em className="italic">{children}</em>
  ),
  del: ({ children }: ComponentPropsWithoutRef<"del">) => (
    <del className="text-[var(--text-tertiary)] line-through">
      {children}
    </del>
  ),
  ul: ({ children }: ComponentPropsWithoutRef<"ul">) => (
    <ul className="my-2 list-disc space-y-1 pl-5 marker:text-[var(--text-quaternary)]">
      {children}
    </ul>
  ),
  ol: ({ children }: ComponentPropsWithoutRef<"ol">) => (
    <ol className="my-2 list-decimal space-y-1 pl-5 marker:text-[var(--text-tertiary)]">
      {children}
    </ol>
  ),
  li: ({ children }: ComponentPropsWithoutRef<"li">) => (
    <li className="pl-0.5">{children}</li>
  ),
  blockquote: ({ children }: ComponentPropsWithoutRef<"blockquote">) => (
    <blockquote className="my-3 border-l-2 border-[var(--accent)] bg-[var(--accent-soft)] px-3 py-1.5 italic text-[var(--text-secondary)]">
      {children}
    </blockquote>
  ),
  a: ({ children, ...props }: ComponentPropsWithoutRef<"a">) => (
    <a
      {...props}
      target="_blank"
      rel="noreferrer"
      className="font-medium text-[var(--accent)] underline decoration-[rgba(12,140,233,0.3)] underline-offset-2 hover:decoration-[var(--accent)]"
    >
      {children}
    </a>
  ),
  hr: () => <hr className="my-4 border-[var(--border-default)]" />,
  pre: ({ children }: ComponentPropsWithoutRef<"pre">) => (
    <pre className="my-3 max-w-full overflow-x-auto rounded-md border border-[var(--border-subtle)] bg-[var(--surface-2)] p-3 font-mono text-[10px] leading-relaxed text-[var(--text-primary)] [&>code]:bg-transparent [&>code]:p-0">
      {children}
    </pre>
  ),
  code: ({
    className,
    children,
    ...props
  }: ComponentPropsWithoutRef<"code">) => (
    <code
      {...props}
      className={cn(
        "rounded bg-[var(--surface-2)] px-1 py-0.5 font-mono text-[10px] text-[var(--text-primary)]",
        className,
      )}
    >
      {children}
    </code>
  ),
  table: ({ children }: ComponentPropsWithoutRef<"table">) => (
    <table className="my-3 block max-w-full overflow-x-auto border-collapse text-left text-[10px]">
      {children}
    </table>
  ),
  th: ({ children }: ComponentPropsWithoutRef<"th">) => (
    <th className="whitespace-nowrap border border-[var(--border-default)] bg-[var(--surface-2)] px-2 py-1 font-semibold text-[var(--text-primary)]">
      {children}
    </th>
  ),
  td: ({ children }: ComponentPropsWithoutRef<"td">) => (
    <td className="border border-[var(--border-default)] px-2 py-1 align-top">
      {children}
    </td>
  ),
  img: ({ className, alt, ...props }: ComponentPropsWithoutRef<"img">) => (
    <img
      {...props}
      alt={alt ?? ""}
      className={cn("my-3 h-auto max-w-full rounded-md", className)}
    />
  ),
};

export function MarkdownArticle({
  text,
  className,
}: {
  text: string;
  className?: string;
}): JSX.Element {
  return (
    <article
      className={cn(
        "min-w-0 text-[11px] leading-[1.7] text-[var(--text-secondary)]",
        className,
      )}
    >
      <Markdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {text}
      </Markdown>
    </article>
  );
}
