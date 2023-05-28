| **Variavel**            | **Tipo**    | **Descrição**                                              |
| :-------------------------- | :---------- | :----------------------------------------------------------- |
| ADR                         | Float       | Taxa Média Diaria. Calculado dividindo a soma de todas as transações de hospedagem pelo número total de estadiasnights. |
| Adults                      | Integer     | Numero de Adultos.                                            |
| Agent                       | Categorical | ID da Agencia de Viagens que foi reservado.               |
| ArrivalDateDayOfMonth       | Integer     | Dia do mês da data de viagem.                        |
| ArrivalDateMonth            | Categorical | Mês da viagem: “Janeiro” até “Dezembro”. |
| ArrivalDateWeekNumber       | Integer     | Semana do Ano da viagem.                             |
| ArrivalDateYear             | Integer     | Ano da viagem.                                        |
| AssignedRoomType            | Categorical | Código do tipo de quarto atribuído à reserva. Às vezes, o tipo de quarto atribuído difere do tipo de quarto reservado devido a razões de operação do hotel (por exemplo, overbooking) ou por solicitação do cliente. |
| Babies                      | Integer     | Numero de bebês.                                            |
| BookingChanges              | Integer     | Número de tentativas/alterações efetuadas à reserva desde o momento em que a reserva foi deferida no sistema até ao momento do check-in ou cancelamento. Calculado adicionando o número de iterações únicas que alteram alguns dos atributos da reserva, nomeadamente: pessoas, data de chegada, noites, tipo de quarto reservado ou refeição.  |
| Children                    | Integer     | Numero de crianças. |
| Company                     | Categorical | ID da empresa/companhia que fez a reserva ou responsável pelo pagamento da reserva. |
| Country                     | Categorical | País de Origem. As categorias são representadas no formato International Standards Organization (ISO) 3155–3:2013. |
| CustomerType                | Categorical | Tipo de reserva, assumindo uma de quatro categorias: Contrato (quando a reserva tem um loteamento ou outro tipo de contrato a ela associado), Grupo (quando a reserva está associada a um grupo), Transitória (quando a reserva não faz parte de um grupo ou contrato, e não está associado a outra reserva transitória) e Transitório-parte (quando a reserva é transitória, mas está associada a pelo menos outra reserva transitória). |
| DaysInWaitingList           | Integer     | Nnúmero de dias que a reserva ficou na lista de espera antes de ser confirmada para o cliente. Calculado subtraindo a data de confirmação da reserva ao cliente à data de entrada da reserva no sistema.  |
| DepositType                 | Categorical | Indicação se o cliente fez um depósito para garantir a reserva. Esta variável pode assumir três categorias: Sem Deposit (não foi efetuado qualquer depósito), Sem Reembolso (foi efetuado um depósito no valor do custo total da estadia) e Reembolsável (foi efetuado um depósito com um valor inferior ao custo total da estadia) . |
| DistributionChannel         | Categorical | Canal de distribuição de reservas. O termo “TA” significa “Agentes de Viagens” e “TO” significa “Operadores Turísticos”. |
| IsCanceled                  | Integer     | Valor indica se a reserva foi cancelada (1) ou não (0). |
| IsRepeatedGuest             | Integer     | Valor indica se a reserva é de um hóspede repetido (1) ou não (0).  |
| LeadTime                    | Integer     | Número de dias decorridos entre a data de reserva e o check-in. |
| MarketSegment               | Categotical | Prestadores de Serviços. Estes estão dividos em categorias entre elas: Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Undefined', 'Aviation'. |
| Meals                       | Categorical | Tipo de Refeição. As categorias são apresentadas em pacotes de refeições padrão de hotelaria: Indefinido/SC (sem pacote de refeições), BB (café da manha e jantar), HB (café da manhã e outra refeição – geralmente jantar) e FB ( café da manhã, almoço e jantar). |
| PreviousBookingsNotCanceled | Integer     | Número de reservas anteriores não canceladas pelo cliente antes da reserva atual. Caso não exista perfil de cliente associado à reserva, o valor é definido como 0. Caso contrário, o valor é o número de reservas com o mesmo perfil de cliente criadas antes da reserva atual e não canceladas. |
| PreviousCancellations       | Integer     | úmero de reservas anteriores que foram canceladas pelo cliente antes da reserva atual. Caso não exista perfil de cliente associado à reserva, o valor é definido como 0. Caso contrário, o valor é o número de reservas com o mesmo perfil de cliente criadas antes da reserva atual e canceladas.  |
| RequiredCarParkingSpaces    | Integer     | Numero de vagas de estacionamentos requisitados.   |
| ReservationStatus           | Categorical | Último estado da reserva, assumindo uma de três categorias: Cancelada (a reserva foi cancelada pelo cliente), Check-Out (o cliente fez o check-in mas já partiu), No-Show (o cliente não fez o check-in e informou o hotel do razão pela qual). |
| ReservationStatusDate       | Date        | Data em que o último status foi definido. Esta variável pode ser usada em conjunto com o `ReservationStatus` para entender quando a reserva foi cancelada ou quando o cliente fez o check-out do hotel. |
| ReservedRoomType            | Categorical | Código do tipo de quarto reservado. |
| StaysInWeekendNights        | Integer     | Número de noites de fim de semana (sábado ou domingo) que o hóspede ficou ou reservou para ficar no hotel. |
| StaysInWeekNights           | Integer     | Número de noites da semana (segunda a sexta) que o hóspede ficou ou reservou para ficar no hotel. |
| TotalOfSpecialRequests      | Integer     | Número de requisições especiais solicidadas pelo hospede (ex: cama de casal, vista para o mar, etc). |