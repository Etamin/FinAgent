
const express = require("express");
const sqlite3 = require("sqlite3").verbose();
const app = express();
const PORT = 3001;

app.use(express.json());

const db = new sqlite3.Database("./bank_api_data.db");

// Core Endpoints (Same as before) ...

// Get customer by ID
app.get("/customers/:id", (req, res) => {
    db.get("SELECT * FROM customers WHERE id = ?", [req.params.id], (err, row) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(row);
    });
});

// Get credit cards by customer ID
app.get("/credit-cards/:customerId", (req, res) => {
    db.all("SELECT * FROM credit_cards WHERE customer_id = ?", [req.params.customerId], (err, rows) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(rows);
    });
});

// Get credit card balance
app.get("/credit-cards/balance/:cardId", (req, res) => {
    db.get("SELECT balance, credit_limit FROM credit_cards WHERE id = ?", [req.params.cardId], (err, row) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(row);
    });
});

// Credit card statement
app.get("/credit-cards/statements/:cardId", (req, res) => {
    const dummyTransactions = [
        { date: "2024-04-01", amount: -120.50, description: "Grocery Store" },
        { date: "2024-03-28", amount: -50.00, description: "Gas Station" },
        { date: "2024-03-25", amount: -200.00, description: "Online Purchase" },
        { date: "2024-03-22", amount: -15.99, description: "Streaming Service" },
        { date: "2024-03-18", amount: 500.00, description: "Payment Received" }
    ];
    res.json({ card_id: req.params.cardId, transactions: dummyTransactions });
});

// Credit card spending
app.post("/credit-cards/spend", (req, res) => {
    const { card_id, amount } = req.body;
    db.run("UPDATE credit_cards SET balance = balance + ? WHERE id = ?", [amount, card_id], function (err) {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ message: "Transaction processed." });
    });
});

// Credit card payment
app.post("/credit-cards/payment", (req, res) => {
    const { card_id, amount } = req.body;
    db.run("UPDATE credit_cards SET balance = balance - ? WHERE id = ?", [amount, card_id], function (err) {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ message: "Payment received." });
    });
});

// Loans
app.get("/loans/:customerId", (req, res) => {
    db.all("SELECT * FROM loans WHERE customer_id = ?", [req.params.customerId], (err, rows) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(rows);
    });
});

app.get("/loan/:loanId", (req, res) => {
    db.get("SELECT * FROM loans WHERE id = ?", [req.params.loanId], (err, row) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(row);
    });
});

app.post("/loan", (req, res) => {
    const { customer_id, loan_type, principal, remaining_balance, due_date } = req.body;
    db.run(
        "INSERT INTO loans (customer_id, loan_type, principal, remaining_balance, due_date) VALUES (?, ?, ?, ?, ?)",
        [customer_id, loan_type, principal, remaining_balance, due_date],
        function (err) {
            if (err) return res.status(500).json({ error: err.message });
            res.json({ loan_id: this.lastID });
        }
    );
});

// Loan payment
app.post("/loan/payment", (req, res) => {
    const { loan_id, amount } = req.body;
    db.run("UPDATE loans SET remaining_balance = remaining_balance - ? WHERE id = ?", [amount, loan_id], function (err) {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ message: "Loan payment processed." });
    });
});

app.listen(PORT, () => {
    console.log(`Bank API Server running at http://localhost:${PORT}`);
});
