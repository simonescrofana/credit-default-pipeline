output "alb_dns_name" {
  description = "Public hostname of the load balancer. Visit http://<value> for the UI, http://<value>:8000/docs for the API."
  value       = aws_lb.main.dns_name
}
