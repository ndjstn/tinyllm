# TinyLLM Alerting Rules Guide

Comprehensive alerting rules for Service Level Objectives (SLOs) and operational monitoring.

## Overview

The alerting rules implement Google SRE best practices for SLO monitoring:
- Multi-window burn rate alerts for rapid error budget consumption detection
- Error budget tracking and depletion warnings
- Latency percentile monitoring with degradation detection
- Distributed tracing health checks

## SLO Targets

### Availability SLO: 99.9%
- **Error Budget**: 0.1% error rate (43.2 minutes of downtime per month)
- **Measurement**: HTTP 5xx errors / total requests
- **Windows**: 5m, 1h, 6h, 30d

### Latency SLO
- **P95**: < 500ms (95% of requests)
- **P99**: < 5s (99% of requests)
- **Measurement**: Request duration from start to completion

### Cache Performance
- **Hit Rate**: > 60%
- **Measurement**: Cache hits / (hits + misses)

## Alert Groups

### 1. SLO Alerts (`tinyllm_slo_alerts`)

#### HighErrorRate
- **Severity**: Critical
- **Threshold**: Error rate > 0.1% for 5 minutes
- **Impact**: Direct SLO breach
- **Action**: Immediate investigation required
- **Typical Causes**:
  - Model API failures
  - Upstream service issues
  - Configuration errors
  - Resource exhaustion

#### HighLatencyP95 / HighLatencyP99
- **Severity**: Warning (P95), Critical (P99)
- **Thresholds**: P95 > 2s, P99 > 5s
- **Impact**: User experience degradation
- **Action**: Investigate slow requests
- **Typical Causes**:
  - Model inference slowdown
  - Database query performance
  - High concurrency
  - Memory pressure

#### CircuitBreakerOpen
- **Severity**: Warning
- **Condition**: Circuit breaker state = OPEN
- **Impact**: Requests to model are being rejected
- **Action**: Check model health, review error logs
- **Typical Causes**:
  - Repeated model failures
  - Network connectivity issues
  - Model server downtime

#### LowCacheHitRate
- **Severity**: Warning
- **Threshold**: Hit rate < 60% for 10 minutes
- **Impact**: Increased latency and costs
- **Action**: Review cache configuration, TTL settings
- **Typical Causes**:
  - Cache too small
  - TTL too short
  - Highly varied query patterns
  - Cache cold start

### 2. Error Budget Alerts (`tinyllm_error_budget`)

#### CriticalBurnRate
- **Severity**: Critical
- **Condition**: Error rate > 14x budget in both 1h and 5m windows
- **Budget Exhaustion**: < 2 days at current rate
- **Action**: IMMEDIATE - Stop deployments, rollback if recent deploy
- **Response Time**: < 15 minutes

#### FastBurnRate
- **Severity**: Warning
- **Condition**: Error rate > 6x budget in both 6h and 30m windows
- **Budget Exhaustion**: < 6 days at current rate
- **Action**: Investigate and plan remediation
- **Response Time**: < 1 hour

#### SlowErrorBudgetBurn
- **Severity**: Warning
- **Condition**: Error rate > 2x budget over 6h
- **Budget Exhaustion**: Depleting faster than planned
- **Action**: Monitor closely, investigate if continues
- **Response Time**: < 4 hours

#### ErrorBudgetDepleting
- **Severity**: Warning
- **Condition**: < 25% error budget remaining in 30-day window
- **Impact**: Limited room for incidents
- **Action**: Avoid risky changes, defer non-critical deploys
- **Response Time**: Plan carefully

### 3. Latency SLO Alerts (`tinyllm_latency_slo`)

#### LatencySLOBreach
- **Severity**: Warning
- **Condition**: < 95% requests under 500ms
- **Impact**: Latency SLO not met
- **Action**: Identify slow endpoints, optimize queries
- **Typical Causes**:
  - Large prompt/response sizes
  - Cold model start
  - Resource contention
  - Inefficient graph execution

#### ModelLatencyDegradation
- **Severity**: Warning
- **Condition**: Model P95 latency 1.5x higher than 1 hour ago
- **Impact**: Performance regression detected
- **Action**: Compare with previous deployments, check model health
- **Typical Causes**:
  - New model version
  - Increased load
  - Hardware issues
  - Configuration changes

### 4. Correlation & Tracing Alerts (`tinyllm_correlation_alerts`)

#### OrphanedCorrelationIDs
- **Severity**: Warning
- **Condition**: > 10/sec correlation IDs not completed
- **Impact**: Incomplete request traces
- **Action**: Check for request timeouts, connection drops
- **Typical Causes**:
  - Client disconnections
  - Request timeouts
  - Trace propagation failures
  - Missing completion handlers

#### LowTraceSamplingRate
- **Severity**: Info
- **Condition**: < 1% of traces sampled
- **Impact**: May miss error scenarios in traces
- **Action**: Verify sampling configuration is intentional
- **Note**: Low sampling is often intentional for high volume

### 5. System Alerts (`tinyllm_system_alerts`)

#### PrometheusScrapeFailure
- **Severity**: Critical
- **Condition**: Cannot scrape metrics from TinyLLM
- **Impact**: No monitoring visibility
- **Action**: Check service health, metrics endpoint
- **Typical Causes**:
  - Service down
  - Network issues
  - Metrics endpoint error
  - Prometheus misconfiguration

#### HighMetricCardinality
- **Severity**: Warning
- **Condition**: > 1000 request series or > 500 error series
- **Impact**: Prometheus performance degradation
- **Action**: Review label usage, implement cardinality controls
- **Typical Causes**:
  - User IDs in labels
  - Unbounded label values
  - High unique graph/model combinations

## Alert Severity Levels

### Critical
- **Response Time**: < 15 minutes
- **Examples**: SLO breach, critical burn rate, service down
- **Escalation**: Page on-call engineer immediately
- **Impact**: Active customer impact or imminent

### Warning
- **Response Time**: < 1 hour
- **Examples**: Elevated error rate, latency degradation
- **Escalation**: Slack notification, create incident
- **Impact**: Potential customer impact if not addressed

### Info
- **Response Time**: Business hours
- **Examples**: Configuration notices, optimization opportunities
- **Escalation**: Log for review
- **Impact**: No immediate customer impact

## Runbook: SLO Incident Response

### Step 1: Assess Severity (< 2 minutes)
```bash
# Check current error rate
curl -s http://localhost:9090/api/v1/query?query=rate(tinyllm_errors_total[5m]) | jq

# Check error budget remaining
curl -s http://localhost:9090/api/v1/query?query='1 - (sum(rate(tinyllm_requests_total{status!~"5.."}[30d])) / sum(rate(tinyllm_requests_total[30d])))' | jq

# Review recent deployments
kubectl rollout history deployment/tinyllm
```

### Step 2: Stop the Bleeding (< 5 minutes)
```bash
# If recent deployment suspected, rollback immediately
kubectl rollout undo deployment/tinyllm

# If specific model causing issues, disable via circuit breaker
curl -X POST http://localhost:8080/admin/circuit-breaker/model-name/open

# Enable enhanced logging for debugging
curl -X POST http://localhost:8080/admin/log-level -d '{"level": "DEBUG"}'
```

### Step 3: Investigate Root Cause (< 30 minutes)
```bash
# Check error logs with correlation IDs
kubectl logs -l app=tinyllm --tail=1000 | grep ERROR

# Review distributed traces
# Open Jaeger/Grafana tempo for recent error traces

# Check model health
curl http://localhost:8080/health/models

# Review metrics in Grafana
# Dashboard: TinyLLM - SLO Monitoring
```

### Step 4: Remediate (Time varies)
- Apply fix (code, config, or infrastructure)
- Test in staging environment
- Deploy carefully with monitoring
- Verify error rate returns to normal

### Step 5: Post-Incident (< 24 hours)
- Create post-mortem document
- Identify preventive measures
- Update runbooks if needed
- Schedule action items

## Configuration

### Adjusting Thresholds
Edit `/docker/prometheus/alerts.yml`:

```yaml
# Example: Change error rate threshold
- alert: HighErrorRate
  expr: |
    (rate(tinyllm_errors_total[5m]) / rate(tinyllm_requests_total[5m])) > 0.001
    # Change 0.001 to your desired threshold
```

### Testing Alerts
```bash
# Validate alert rules
promtool check rules /docker/prometheus/alerts.yml

# Test specific alert expression
curl -s 'http://localhost:9090/api/v1/query?query=rate(tinyllm_errors_total[5m]) > 0.001' | jq
```

### Silencing Alerts
```bash
# Silence specific alert during maintenance
amtool silence add \
  alertname=HighErrorRate \
  --duration=1h \
  --comment="Planned maintenance"
```

## Integration

### Slack Notifications
Configure Alertmanager webhook in `alertmanager.yml`:
```yaml
receivers:
  - name: 'slack-critical'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#alerts-critical'
        title: 'TinyLLM Alert: {{ .GroupLabels.alertname }}'
```

### PagerDuty Integration
```yaml
receivers:
  - name: 'pagerduty-oncall'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        severity: '{{ .CommonLabels.severity }}'
```

## Best Practices

1. **Don't Alert on Everything**: Alert only on customer-impacting conditions
2. **Multi-Window Alerts**: Use multiple time windows to reduce false positives
3. **Actionable Alerts**: Every alert must have a clear action item
4. **Error Budget Philosophy**: Accept some errors; focus on trends
5. **Regular Review**: Update thresholds based on actual system behavior
6. **Test Alerts**: Periodically trigger test alerts to verify notification pipeline

## Metrics Reference

### Key Metrics
- `tinyllm_requests_total`: Total request count by status
- `tinyllm_errors_total`: Error count by type
- `tinyllm_request_duration_seconds`: Request latency histogram
- `tinyllm_circuit_breaker_state`: Circuit breaker status (0=closed, 1=half-open, 2=open)
- `tinyllm_cache_hits_total` / `tinyllm_cache_misses_total`: Cache performance
- `tinyllm_correlation_ids_total`: Correlation ID creation rate
- `tinyllm_model_latency_ms`: Per-model latency tracking

## Resources

- [Google SRE Book - Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/)
- [Prometheus Alerting Best Practices](https://prometheus.io/docs/practices/alerting/)
- [Grafana Dashboard: TinyLLM SLO Monitoring](http://localhost:3000/d/tinyllm-slo)

## Support

For questions or issues with alerting:
1. Check Grafana dashboards for current metrics
2. Review Prometheus alert status: http://localhost:9090/alerts
3. Check Alertmanager status: http://localhost:9093
4. Review logs: `kubectl logs -l app=tinyllm -c prometheus`
