# Incident Postmortem — Production Database Outage

| Field    | Details                       |
|----------|-------------------------------|
| Date     | 7 November 2023               |
| Severity | SEV-1 (full service outage)   |
| Duration | 2 hours 14 minutes            |
| Author   | DevOps / SRE Team             |

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 14:02 | Automated monitoring alerts fire: API error rate spikes to 94%. |
| 14:04 | On-call engineer (David Reyes) paged via PagerDuty. |
| 14:07 | David joins incident bridge call. Initial hypothesis: upstream dependency. |
| 14:15 | Database team joins. RDS PostgreSQL primary shows CPU at 100%. |
| 14:22 | Root cause identified: runaway query from a missing index on `orders` table. |
| 14:35 | Decision made to kill long-running queries and add index concurrently. |
| 14:51 | Index build begins (concurrent, non-blocking). |
| 16:04 | Index build complete. CPU drops to 12%. Error rate returns to <0.1%. |
| 16:16 | All-clear declared. Incident closed. |

## Root Cause

A new analytics query deployed in the 13:45 UTC release queried the `orders` table
without using any index, resulting in sequential scans on a **180-million row** table.
Under normal load this took 45 seconds. Under peak traffic (Black Tuesday sale event)
it saturated all database connections in the pool within minutes.

The query was:

```sql
SELECT user_id, SUM(total) FROM orders WHERE status = 'completed' GROUP BY user_id;
```

The `orders` table had no index on the `status` column. A full sequential scan was performed
for every request, and connection pool exhaustion caused cascading failures in the API layer.

## Contributing Factors

1. The analytics query was not reviewed by the database team before deployment.
2. Staging database only has 50,000 rows — the missing index wasn't caught in testing.
3. No query performance regression check in the CI/CD pipeline.
4. Connection pool size (50) was too small for the traffic spike during the sale event.

## Impact

- ~14,000 failed user requests during the outage window.
- Estimated revenue loss: **$38,000**.
- Customer support ticket volume increased 4x for 24 hours post-incident.

## Action Items

- [x] Add index on `orders.status` column.
- [ ] Increase RDS connection pool from 50 to 200 (requires instance resize). *(IN PROGRESS)*
- [ ] Add `pg_stat_statements` monitoring to catch slow queries automatically.
- [ ] Require DBA sign-off for any query touching tables >10M rows.
- [ ] Seed staging database with production-scale data (anonymised).
- [ ] Add automated `EXPLAIN ANALYZE` check in CI for new SQL queries.

## Lessons Learned

Scale testing with realistic data volumes is non-negotiable for database queries.
Our deployment checklist must include a database impact section.
