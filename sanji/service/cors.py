"""Credentialed cross-origin CORS middleware (#81).

sandjig's ``add_cors_headers`` after_request sets ``Access-Control-Allow-Origin: *``
unconditionally. Flask runs ``after_request`` hooks in reverse registration order,
so sandjig's (registered during ``create_app``) runs *last* — a sanji-level
``after_request`` cannot override it. Browsers reject a wildcard
``Access-Control-Allow-Origin`` on any request sent with credentials (cookies), so
the cookie-authenticated SPA cannot call the API cross-origin.

This WSGI middleware runs *after* the entire Flask request cycle (including
sandjig's hook), so it has the final say on the response headers. For a request
whose ``Origin`` is in the configured allowlist it replaces the wildcard with that
specific origin and adds ``Access-Control-Allow-Credentials: true``; a request from
any other origin is denied CORS access (the wildcard is stripped).
"""

from collections.abc import Callable, Iterable

_ALLOW_ORIGIN = "Access-Control-Allow-Origin"
_ALLOW_CREDENTIALS = "Access-Control-Allow-Credentials"
_ALLOW_METHODS = "Access-Control-Allow-Methods"
_VARY = "Vary"

# Explicit method list — a credentialed response may not use the ``*`` wildcard
# that sandjig sets for ``Access-Control-Allow-Methods``.
_CREDENTIALED_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS"

_Header = tuple[str, str]
_Headers = list[_Header]


class CorsCredentialsMiddleware:
    """Rewrite CORS response headers to support allow-listed credentialed origins."""

    def __init__(self, wsgi_app: Callable, allowed_origins: frozenset[str]) -> None:
        self._wsgi_app = wsgi_app
        self._allowed_origins = allowed_origins

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        request_origin = environ.get("HTTP_ORIGIN")

        def cors_start_response(
            status: str, headers: _Headers, exc_info=None
        ) -> Callable:
            rewritten = self._rewrite(list(headers), request_origin)
            return start_response(status, rewritten, exc_info)

        return self._wsgi_app(environ, cors_start_response)

    def _rewrite(self, headers: _Headers, request_origin: str | None) -> _Headers:
        # A request without an Origin header is not a CORS request — leave it alone.
        if not request_origin:
            return headers

        managed = {_ALLOW_ORIGIN.lower(), _ALLOW_CREDENTIALS.lower()}
        result = [(k, v) for (k, v) in headers if k.lower() not in managed]

        # Strip the permissive wildcard method grant regardless of allow/deny.
        result = [(k, v) for (k, v) in result if k.lower() != _ALLOW_METHODS.lower()]

        if request_origin not in self._allowed_origins:
            # Denied: no CORS grant is emitted for a non-allowlisted origin.
            return result

        result.append((_ALLOW_ORIGIN, request_origin))
        result.append((_ALLOW_CREDENTIALS, "true"))
        result.append((_ALLOW_METHODS, _CREDENTIALED_METHODS))
        return self._with_vary_origin(result)

    @staticmethod
    def _with_vary_origin(headers: _Headers) -> _Headers:
        """Add ``Origin`` to ``Vary`` (merging any existing values) so shared caches
        do not serve one origin's CORS headers to another."""
        others = [(k, v) for (k, v) in headers if k.lower() != _VARY.lower()]
        values: list[str] = []
        for key, value in headers:
            if key.lower() == _VARY.lower():
                values.extend(p.strip() for p in value.split(",") if p.strip())
        if "Origin" not in values:
            values.append("Origin")
        others.append((_VARY, ", ".join(values)))
        return others
