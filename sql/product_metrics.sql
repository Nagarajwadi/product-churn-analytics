WITH user_signup AS (
    SELECT
        user_id,
        MIN(timestamp) AS signup_date
    FROM clean_events
    WHERE event_name = 'signup'
    GROUP BY user_id
),

eligible_users AS (
    SELECT user_id
    FROM user_signup
    WHERE signup_date <= '2026-04-30'
),

feature_metrics AS (
    SELECT
        e.user_id,

        COUNT(*) AS total_events,
        COUNT(DISTINCT e.session_id) AS total_sessions,
        COUNT(DISTINCT DATE(e.timestamp)) AS active_days,

        SUM(CASE WHEN e.event_name = 'login' THEN 1 ELSE 0 END) AS login_count,
        SUM(CASE WHEN e.event_name = 'view_product' THEN 1 ELSE 0 END) AS product_view_count,
        SUM(CASE WHEN e.event_name = 'search' THEN 1 ELSE 0 END) AS search_count,
        SUM(CASE WHEN e.event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS add_to_cart_count,
        SUM(CASE WHEN e.event_name = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
        SUM(CASE WHEN e.event_name = 'subscription' THEN 1 ELSE 0 END) AS subscription_count

    FROM clean_events e
    INNER JOIN eligible_users u
        ON e.user_id = u.user_id

    WHERE e.timestamp <= '2026-04-30 23:59:59'

    GROUP BY e.user_id
),

churn_outcome AS (
    SELECT DISTINCT
        e.user_id
    FROM clean_events e
    INNER JOIN eligible_users u
        ON e.user_id = u.user_id

    WHERE e.event_name = 'cancel_subscription'
      AND e.timestamp >= '2026-05-01 00:00:00'
      AND e.timestamp <= '2026-06-30 23:59:59'
)

SELECT
    f.user_id,

    f.total_events,
    f.total_sessions,
    f.active_days,

    f.login_count,
    f.product_view_count,
    f.search_count,
    f.add_to_cart_count,
    f.purchase_count,
    f.subscription_count,

    -- Behavioral features
    ROUND(
        CAST(f.total_events AS REAL) /
        NULLIF(f.active_days, 0),
        2
    ) AS events_per_active_day,

    ROUND(
        CAST(f.total_sessions AS REAL) /
        NULLIF(f.active_days, 0),
        2
    ) AS sessions_per_active_day,

    ROUND(
        CAST(f.search_count AS REAL) /
        NULLIF(f.total_events, 0),
        4
    ) AS search_rate,

    ROUND(
        CAST(f.add_to_cart_count AS REAL) /
        NULLIF(f.total_events, 0),
        4
    ) AS cart_rate,

    ROUND(
        CAST(f.purchase_count AS REAL) /
        NULLIF(f.total_events, 0),
        4
    ) AS purchase_rate,

    CASE
        WHEN c.user_id IS NOT NULL THEN 1
        ELSE 0
    END AS churn

FROM feature_metrics f

LEFT JOIN churn_outcome c
    ON f.user_id = c.user_id;