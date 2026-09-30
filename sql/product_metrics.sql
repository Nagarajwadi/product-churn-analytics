-- ==================================================
-- PRODUCT METRICS + CHURN LABEL
-- ==================================================

SELECT
    user_id,

    COUNT(*) AS total_events,

    SUM(
        CASE
            WHEN event_name = 'login'
            THEN 1
            ELSE 0
        END
    ) AS login_count,

    SUM(
        CASE
            WHEN event_name = 'view_product'
            THEN 1
            ELSE 0
        END
    ) AS product_view_count,

    SUM(
        CASE
            WHEN event_name = 'search'
            THEN 1
            ELSE 0
        END
    ) AS search_count,

    SUM(
        CASE
            WHEN event_name = 'add_to_cart'
            THEN 1
            ELSE 0
        END
    ) AS add_to_cart_count,

    SUM(
        CASE
            WHEN event_name = 'purchase'
            THEN 1
            ELSE 0
        END
    ) AS purchase_count,

    SUM(
        CASE
            WHEN event_name = 'subscription'
            THEN 1
            ELSE 0
        END
    ) AS subscription_count,

    CASE
        WHEN SUM(
            CASE
                WHEN event_name = 'cancel_subscription'
                THEN 1
                ELSE 0
            END
        ) > 0
        THEN 1
        ELSE 0
    END AS churn

FROM clean_events

GROUP BY user_id;