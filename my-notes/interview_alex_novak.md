# Candidate Interview Notes — Senior Backend Engineer

| Field       | Details             |
|-------------|---------------------|
| Date        | 14 February 2024    |
| Interviewer | Priya Sharma        |
| Candidate   | Alex Novak          |

## Round 1 — Technical Screen (60 min)

### Coding Problem: Design a rate limiter supporting sliding window algorithm

**Performance:** Alex implemented a Redis-based solution using sorted sets.
Time complexity was correct (O(log N)). Code was clean and well-commented.
Edge cases handled: burst traffic, distributed instances, clock skew.

**Score: 9/10**

### System Design: Design a URL shortener like bit.ly

**Covered:** hashing strategy (Base62), database schema, cache layer (Redis),
CDN for redirect latency, analytics tracking, and horizontal scaling.

**Missed:** discussing write-ahead logging and failure recovery for the DB.

**Score: 7/10**

## Round 2 — Behavioural (45 min)

- **Conflict resolution:** Good answer about disagreeing with a tech lead over
  microservices vs monolith. Demonstrated data-driven persuasion skills.
- **Ownership:** Led migration of a Django app to FastAPI, reduced p99 latency by 40%.
- **Communication:** Clear and concise. Explains complex topics well.

## Overall Feedback

Strong technical fundamentals. Python and Go expertise matches our stack.
Slightly weaker on distributed systems depth but eager to learn.
Cultural fit: excellent — collaborative, humble, growth-minded.

**Recommendation:** HIRE — proceed to offer stage.
**Suggested level:** L4 (Senior I). **Salary band:** $160k–$180k base.
