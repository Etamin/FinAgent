
# 📘 Bank API Server – Full Documentation

Base URL: `http://localhost:3001`

---

## 👤 Customers

### GET `/customers/:id`
Get customer details by ID.

---

## 💳 Credit Cards

### GET `/credit-cards/:customerId`
Get all credit cards for a customer.

### GET `/credit-cards/balance/:cardId`
Check the balance and credit limit of a card.

### GET `/credit-cards/statements/:cardId`
Returns a mock list of recent transactions (statement).

### POST `/credit-cards/spend`
Simulate a purchase (increases balance).

**Request Body**:
```json
{
  "card_id": 1,
  "amount": 100.00
}
```

### POST `/credit-cards/payment`
Simulate a payment (reduces balance).

**Request Body**:
```json
{
  "card_id": 1,
  "amount": 150.00
}
```

---

## 💰 Loans

### GET `/loans/:customerId`
Get all loans for a customer.

### GET `/loan/:loanId`
Get a specific loan by ID.

### POST `/loan`
Create a new loan.

**Request Body**:
```json
{
  "customer_id": 2,
  "loan_type": "education",
  "principal": 20000,
  "remaining_balance": 20000,
  "due_date": "2027-06-30"
}
```

### POST `/loan/payment`
Make a payment toward an existing loan.

**Request Body**:
```json
{
  "loan_id": 3,
  "amount": 2500
}
```

---

## ✅ Notes

- All credit card and loan balances are managed by direct numeric updates.
- There is no authentication in this demo server.
- Useful for simulating a bank backend for chatbot or agent integration.
