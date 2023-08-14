run:
	docker-compose up --build -d
stop:
	docker-compose down 
rm:
	docker rmi $(docker images -q)
