-- API-enriched engagement analysis
--
-- Purpose:
-- Compare product engagement across departments
-- for users enriched with external API data.
--
-- Note:
-- API data is synthetic/demo data and is available
-- for only 208 of the 5,000 product users.

SELECT
    department,
    COUNT(*) AS users,
    ROUND(AVG(event_count), 2) AS avg_events,
    MIN(event_count) AS min_events,
    MAX(event_count) AS max_events
FROM user_api_enrichment
GROUP BY department
ORDER BY avg_events DESC;
