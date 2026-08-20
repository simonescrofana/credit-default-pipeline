output "alb_dns_name" {
  description = "Public hostname of the load balancer. Visit http://<value> for the UI, http://<value>:8000/docs for the API."
  value       = aws_lb.main.dns_name
}

output "cloudfront_domain_name" {
  description = "Public CloudFront domain. Visit https://<value>/ for the UI, https://<value>/api/docs for the API. This is the stable URL to share going forward."
  value       = aws_cloudfront_distribution.main.domain_name
}
