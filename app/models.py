from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class LeasingApplication(Base):
    __tablename__ = "leasing_applications"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="Черновик")
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    lessee_company = Column(String)
    lessee_inn = Column(String)
    lessee_legal_address = Column(String)
    lessee_actual_address = Column(String)
    lessee_director = Column(String)
    asset_term = Column(Integer)
    advance_payment_percent = Column(Integer)

    user = relationship("User", back_populates="applications")
    suppliers = relationship("Supplier", back_populates="application", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="application", cascade="all, delete-orphan")
    guarantors = relationship("Guarantor", back_populates="application", cascade="all, delete-orphan")
    pledges = relationship("Pledge", back_populates="application", cascade="all, delete-orphan")

# ✅ НОВЫЕ МОДЕЛИ ДЛЯ СВЯЗАННЫХ ДАННЫХ
class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("leasing_applications.id"))
    company = Column(String)
    inn = Column(String)
    address = Column(String)
    application = relationship("LeasingApplication", back_populates="suppliers")

class Asset(Base):
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("leasing_applications.id"))
    name = Column(String)
    quantity = Column(Integer)
    delivery_time = Column(Integer)
    total_cost = Column(String)
    application = relationship("LeasingApplication", back_populates="assets")

class Guarantor(Base):
    __tablename__ = "guarantors"
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("leasing_applications.id"))
    full_name = Column(String)
    inn = Column(String)
    contacts = Column(String)
    application = relationship("LeasingApplication", back_populates="guarantors")

class Pledge(Base):
    __tablename__ = "pledges"
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("leasing_applications.id"))
    name = Column(String)
    quantity = Column(Integer)
    market_value = Column(String)
    application = relationship("LeasingApplication", back_populates="pledges")

class FileMetadata(Base):
    __tablename__ = "file_metadata"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    table_name = Column(String)
    file_size = Column(Integer)
    upload_time = Column(DateTime, default=datetime.utcnow)

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="chat")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    sender = Column(String)  # 'user' or 'ai'
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")
