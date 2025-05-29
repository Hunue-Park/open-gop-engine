.PHONY: build up down logs local local-with-logs

build:
	docker-compose -p speech-recognition -f docker-compose.yml build

up:
	docker-compose -p speech-recognition -f docker-compose.yml up -d

up-with-logs:
	docker-compose -p speech-recognition -f docker-compose.yml up

down:
	docker-compose -f docker-compose.yml down

logs:
	docker-compose -f docker-compose.yml logs -f

local: build up
	@echo "로컬 개발 환경이 http://localhost:8000 에서 실행 중입니다."

local-with-logs: build up-with-logs