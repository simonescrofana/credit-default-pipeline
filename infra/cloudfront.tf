resource "aws_cloudfront_function" "strip_api_prefix" {
  name    = "${var.project_name}-strip-api-prefix"
  runtime = "cloudfront-js-2.0"
  comment = "Removes the /api prefix before forwarding to the FastAPI origin"
  publish = true
  code    = <<-EOT
    function handler(event) {
      var request = event.request;
      request.uri = request.uri.replace(/^\/api/, "");
      if (request.uri === "") {
        request.uri = "/";
      }
      return request;
    }
  EOT
}

resource "aws_cloudfront_distribution" "main" {
  enabled = true

  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "ui"

    custom_origin_config {
      http_port              = 80
      https_port              = 443
      origin_protocol_policy   = "http-only"
      origin_ssl_protocols     = ["TLSv1.2"]
    }
  }

  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "api"

    custom_origin_config {
      http_port              = 8000
      https_port              = 443
      origin_protocol_policy   = "http-only"
      origin_ssl_protocols     = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = "ui"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods         = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]

    forwarded_values {
      query_string = true
      headers      = ["*"]

      cookies {
        forward = "all"
      }
    }

    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.edge_wake.qualified_arn
      include_body = false
    }
  }

  ordered_cache_behavior {
    path_pattern            = "/api/*"
    target_origin_id        = "api"
    viewer_protocol_policy  = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods            = ["GET", "HEAD"]

    forwarded_values {
      query_string = true
      headers      = ["*"]

      cookies {
        forward = "all"
      }
    }

    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.strip_api_prefix.arn
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
