import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
import datetime
from sqlalchemy.orm import Session

load_dotenv()

# Read connection strings from environment variables
COL_SERVER_CONN = os.getenv('COL_SERVER_CONN')
KYC_SERVER_CONN = os.getenv('KYC_SERVER_CONN')

# Create SQLAlchemy engines
col_engine = create_engine(COL_SERVER_CONN, poolclass=NullPool)
kyc_engine = create_engine(KYC_SERVER_CONN, poolclass=NullPool)

# Session makers
ColSession = sessionmaker(bind=col_engine)
KycSession = sessionmaker(bind=kyc_engine)

Base = declarative_base()

class VideoKYC(Base):
    __tablename__ = 'videokyc'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ssp_id = Column(Integer)
    request_id = Column(String)
    created_on = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String)
    video_url = Column(String)
    image_url = Column(String)
    image_type = Column(String)
    speech_score = Column(Integer)
    face_score = Column(Integer)
    liveness_score = Column(Integer)

# Helper to insert a new KYC record
def insert_kyc_record(
    ssp_id: int,
    request_id: str,
    status: str,
    video_url: str,
    image_url: str,
    image_type: str,
    speech_score: int,
    face_score: int,
    liveness_score: int
):
    session: Session = KycSession()
    try:
        record = VideoKYC(
            ssp_id=ssp_id,
            request_id=request_id,
            status=status,
            video_url=video_url,
            image_url=image_url,
            image_type=image_type,
            speech_score=speech_score,
            face_score=face_score,
            liveness_score=liveness_score
        )
        session.add(record)
        session.commit()
        return record.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def insert_kyc_start_record(ssp_id: int, request_id: str):
    session: Session = KycSession()
    try:
        record = VideoKYC(
            ssp_id=ssp_id,
            request_id=request_id,
            status="started"
        )
        session.add(record)
        session.commit()
        return record.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def update_kyc_record(
    kyc_id: int,
    status: str,
    video_url: str,
    image_url: str,
    image_type: str,
    speech_score: int,
    face_score: int,
    liveness_score: int
):
    session: Session = KycSession()
    try:
        record = session.query(VideoKYC).filter_by(id=kyc_id).first()
        if record:
            record.status = status
            record.video_url = video_url
            record.image_url = image_url
            record.image_type = image_type
            record.speech_score = speech_score
            record.face_score = face_score
            record.liveness_score = liveness_score
            session.commit()
        else:
            raise Exception(f"KYC record with id {kyc_id} not found")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def update_kyc_record_by_ssp_and_request_id(
    ssp_id: int,
    request_id: str,
    status: str,
    video_url: str,
    image_url: str,
    image_type: str,
    speech_score: int,
    face_score: int,
    liveness_score: int
):
    session: Session = KycSession()
    try:
        record = session.query(VideoKYC).filter_by(ssp_id=ssp_id, request_id=request_id).first()
        if record:
            record.status = status
            record.video_url = video_url
            record.image_url = image_url
            record.image_type = image_type
            record.speech_score = speech_score
            record.face_score = face_score
            record.liveness_score = liveness_score
            session.commit()
        else:
            raise Exception(f"KYC record with ssp_id {ssp_id} and request_id {request_id} not found")
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close() 