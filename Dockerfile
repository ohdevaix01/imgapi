FROM golang:1.22-alpine AS build
WORKDIR /src
COPY main.go .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /imgapi main.go

FROM alpine:3.20 AS certs
RUN apk add --no-cache ca-certificates

FROM scratch
COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=build /imgapi /imgapi
COPY public /app/public
ENV PORT=8080 DATA_DIR=/data
EXPOSE 8080
ENTRYPOINT ["/imgapi"]
