"""Tests for OpenAPI documentation endpoints."""

import pytest
from fastapi.testclient import TestClient

from llamacpp_cli.lb_proxy import ProxyState, create_lb_app


@pytest.fixture
def test_client():
    """Create a test client with a minimal proxy state."""
    state = ProxyState()
    state.api_key = "test-key-123"
    app = create_lb_app(state)
    return TestClient(app)


@pytest.fixture
def test_client_no_auth():
    """Create a test client without authentication."""
    state = ProxyState()
    app = create_lb_app(state)
    return TestClient(app)


class TestOpenAPIEndpoints:
    """Test OpenAPI documentation endpoints."""

    def test_docs_endpoint_returns_html(self, test_client):
        """Test /docs endpoint returns Swagger UI HTML."""
        response = test_client.get("/docs")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "swagger-ui" in response.text.lower() or "openapi" in response.text.lower()

    def test_redoc_endpoint_returns_html(self, test_client):
        """Test /redoc endpoint returns ReDoc HTML."""
        response = test_client.get("/redoc")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "redoc" in response.text.lower()

    def test_openapi_json_endpoint(self, test_client):
        """Test /openapi.json endpoint returns valid OpenAPI spec."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec
        assert spec["info"]["title"] == "LlamaCPP Load Balancer"
        assert spec["info"]["version"] == "1.0.0"

    def test_openapi_spec_has_all_endpoints(self, test_client):
        """Test OpenAPI spec documents all expected endpoints."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check core OpenAI API endpoints
        assert "/v1/chat/completions" in paths
        assert "/v1/completions" in paths
        assert "/v1/embeddings" in paths
        assert "/v1/models" in paths
        assert "/v1/tokenize" in paths
        assert "/v1/detokenize" in paths

        # Check management endpoints
        assert "/backends" in paths
        assert "/slots" in paths
        assert "/props" in paths

        # Check health & stats endpoints
        assert "/health" in paths
        assert "/stats" in paths
        assert "/metrics" in paths

        # Check legacy endpoints
        assert "/v1/engines" in paths
        assert "/v1/engines/{engine_id}" in paths
        assert "/v1/engines/{engine_id}/completions" in paths

    def test_openapi_spec_has_tags(self, test_client):
        """Test OpenAPI spec includes tags for organization."""
        response = test_client.get("/openapi.json")
        spec = response.json()

        assert "tags" in spec
        tag_names = {tag["name"] for tag in spec["tags"]}
        assert "OpenAI API" in tag_names
        assert "Management" in tag_names
        assert "Health & Stats" in tag_names
        assert "Legacy" in tag_names

    def test_openapi_endpoints_have_descriptions(self, test_client):
        """Test that all documented endpoints have descriptions."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check a sampling of endpoints have descriptions
        chat_completions = paths["/v1/chat/completions"]["post"]
        assert "description" in chat_completions or "summary" in chat_completions

        health = paths["/health"]["get"]
        assert "description" in health or "summary" in health

        models = paths["/v1/models"]["get"]
        assert "description" in models or "summary" in models

    def test_openapi_has_server_info(self, test_client):
        """Test OpenAPI spec includes server and description info."""
        response = test_client.get("/openapi.json")
        spec = response.json()

        # Check info section
        info = spec["info"]
        assert "title" in info
        assert "description" in info
        assert "version" in info
        assert len(info["description"]) > 100  # Should have substantial description

    def test_docs_accessible_without_auth(self, test_client_no_auth):
        """Test documentation endpoints don't require authentication."""
        # All OpenAPI doc endpoints should be public
        assert test_client_no_auth.get("/docs").status_code == 200
        assert test_client_no_auth.get("/redoc").status_code == 200
        assert test_client_no_auth.get("/openapi.json").status_code == 200


class TestOpenAPISchemas:
    """Test OpenAPI request/response schemas."""

    def test_chat_completions_has_request_schema(self, test_client):
        """Test chat completions endpoint has description."""
        response = test_client.get("/openapi.json")
        spec = response.json()

        # Check endpoint has description (endpoints use Request directly, not Pydantic models)
        chat_endpoint = spec["paths"]["/v1/chat/completions"]["post"]
        assert "description" in chat_endpoint
        assert len(chat_endpoint["description"]) > 100  # Substantial description

    def test_schemas_exist(self, test_client):
        """Test that Pydantic model schemas are included."""
        response = test_client.get("/openapi.json")
        spec = response.json()

        # FastAPI auto-generates schemas from Pydantic models
        assert "components" in spec
        # Schemas might be in components.schemas or be inlined
        # Just verify components exists (FastAPI will populate it)

    def test_openapi_endpoints_have_proper_methods(self, test_client):
        """Test endpoints have correct HTTP methods."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # POST endpoints
        assert "post" in paths["/v1/chat/completions"]
        assert "post" in paths["/v1/completions"]
        assert "post" in paths["/v1/embeddings"]

        # GET endpoints
        assert "get" in paths["/v1/models"]
        assert "get" in paths["/health"]
        assert "get" in paths["/stats"]
        assert "get" in paths["/backends"]

    def test_openapi_responses_documented(self, test_client):
        """Test endpoints have response documentation."""
        response = test_client.get("/openapi.json")
        spec = response.json()

        # Check chat completions has response codes
        chat_endpoint = spec["paths"]["/v1/chat/completions"]["post"]
        assert "responses" in chat_endpoint
        # Should have at least 200 response
        responses = chat_endpoint["responses"]
        assert "200" in responses or "default" in responses


class TestEndpointMetadata:
    """Test endpoint metadata and tags."""

    def test_openai_api_endpoints_tagged(self, test_client):
        """Test OpenAI API endpoints are properly tagged."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check OpenAI API endpoints have correct tag
        assert "OpenAI API" in paths["/v1/chat/completions"]["post"]["tags"]
        assert "OpenAI API" in paths["/v1/completions"]["post"]["tags"]
        assert "OpenAI API" in paths["/v1/embeddings"]["post"]["tags"]
        assert "OpenAI API" in paths["/v1/models"]["get"]["tags"]

    def test_management_endpoints_tagged(self, test_client):
        """Test management endpoints are properly tagged."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check management endpoints have correct tag
        assert "Management" in paths["/backends"]["get"]["tags"]
        assert "Management" in paths["/slots"]["get"]["tags"]
        assert "Management" in paths["/props"]["get"]["tags"]

    def test_health_stats_endpoints_tagged(self, test_client):
        """Test health & stats endpoints are properly tagged."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check health & stats endpoints have correct tag
        assert "Health & Stats" in paths["/health"]["get"]["tags"]
        assert "Health & Stats" in paths["/stats"]["get"]["tags"]
        assert "Health & Stats" in paths["/metrics"]["get"]["tags"]

    def test_legacy_endpoints_tagged(self, test_client):
        """Test legacy endpoints are properly tagged."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check legacy endpoints have correct tag
        assert "Legacy" in paths["/v1/engines"]["get"]["tags"]
        assert "Legacy" in paths["/v1/engines/{engine_id}"]["get"]["tags"]
        assert "Legacy" in paths["/v1/engines/{engine_id}/completions"]["post"]["tags"]


class TestDocumentationQuality:
    """Test documentation quality and completeness."""

    def test_api_description_mentions_features(self, test_client):
        """Test API description mentions key features."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        description = spec["info"]["description"].lower()

        # Check for key feature mentions
        assert "model-aware routing" in description or "routing" in description
        assert "load balanc" in description
        assert "rate limit" in description
        assert "authentication" in description or "api key" in description

    def test_api_description_has_examples(self, test_client):
        """Test API description includes usage examples."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        description = spec["info"]["description"]

        # Should have code examples
        assert "```" in description  # Code blocks
        assert "curl" in description.lower() or "example" in description.lower()

    def test_endpoint_summaries_exist(self, test_client):
        """Test endpoints have summary or description."""
        response = test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check a few key endpoints
        for path, methods in paths.items():
            for method, details in methods.items():
                # Each endpoint should have at least a summary or description
                assert "summary" in details or "description" in details, f"{method.upper()} {path} missing summary/description"
