# Q2 2024 Product Roadmap — Internal Planning Document

**Prepared by:** Product Team
**Last Updated:** 1 April 2024

## Goals for Q2

The primary objective this quarter is to ship the **v2.0 release** of our analytics platform,
focusing on performance improvements, a redesigned dashboard, and new AI-powered features.

## Feature 1 — AI Query Assistant `(Priority: HIGH)`

Allow users to ask natural language questions about their data.
Powered by Claude API with RAG over the user's connected datasets.

- **Estimated effort:** 6 weeks (2 engineers)
- **Target release:** End of May 2024

## Feature 2 — Dashboard Redesign `(Priority: HIGH)`

Current dashboard scores **42/100** on usability testing. Target: **75+**.
Switching from Chart.js to Recharts for better customisation.
New dark mode, drag-and-drop widget layout, and responsive mobile view.

- **Estimated effort:** 4 weeks (1 engineer + 1 designer)

## Feature 3 — Automated Anomaly Detection `(Priority: MEDIUM)`

Integrate a lightweight time-series anomaly detection model (Isolation Forest).
Users receive Slack/email alerts when KPIs deviate by more than 2 standard deviations.

- **Estimated effort:** 3 weeks (1 data scientist)

## Feature 4 — SAML 2.0 SSO `(Priority: MEDIUM)`

Enterprise customers have been requesting SSO for two quarters.
Integrate with Okta, Azure AD, and Google Workspace.

- **Estimated effort:** 2 weeks (1 engineer)

## Risks

- AI Query Assistant depends on finalising the RAG architecture (decision pending).
- Designer resource may be constrained if the onboarding project runs over.

## OKRs

- Ship v2.0 by **30 June 2024**.
- Reduce dashboard load time from **4.2s** to under **1.5s**.
- Achieve NPS score of **50+** by end of Q2 (current: 34).
