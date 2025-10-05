#make restart   it will rebuild + restart everything
restart:
	docker compose down
	docker compose up -d --build

logs:
	docker compose logs -f

down:
	docker compose down