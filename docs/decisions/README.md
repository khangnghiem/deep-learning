# Architecture Decision Records (ADRs)

> Lightweight, append-only records of architectural and infrastructure decisions.
> Follows the **MADR (Markdown Architectural Decision Records)** format.

---

## Decision Log

| ID | Title | Status | Date | Decision-Makers |
|---|---|---|---|---|
| [**`ADR-0001`**](ADR-0001-medallion-data-lake-architecture.md) | Adopt 4-Tier Medallion Architecture with Hierarchical Bronze & Offline Feature Store | `accepted` | 2026-09-07 | `@khangnghiem` |
| [**`ADR-0002`**](ADR-0002-google-drive-three-pillar-layout.md) | Standardize Google Drive on 3-Pillars (`data`, `models`, `ops`) and Centralized `archive/` | `accepted` | 2026-09-07 | `@khangnghiem` |

---

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences.

- **Immutable**: Never edit an accepted ADR to reflect new architecture. Create a new ADR that supersedes the old one and mark the old ADR's status as `superseded`.
- **Naming**: `ADR-NNNN-<slug>.md` (4-digit zero-padded number).
- **Structure**:
  - Frontmatter: `id`, `status` (`proposed` | `accepted` | `superseded` | `rejected`), `date`, `decision-makers`.
  - Sections: `# NNNN — <Title>`, `## Context`, `## Decision`, `## Consequences`.
